import dataclasses
import warnings
from typing import Dict, Generator, Iterator, List, Optional, Union, cast

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

from nanotron import distributed as dist
from nanotron import logging
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.random import set_random_seed
try:
    import datasets
    from datasets import (
        Dataset,
        DatasetDict,
        Features,
        Sequence,
        Value,
        concatenate_datasets,
        load_dataset,
    )
    from transformers import PreTrainedTokenizerBase
    from transformers.trainer_pt_utils import DistributedSamplerWithLoop
except ImportError:
    warnings.warn("Datasets and/or Transformers not installed, you'll be unable to use the dataloader.")

from nanotron.config import (
    Config,
    DataArgs,
    DatasetStageArgs,
    MultilingualNanosetDatasetsArgs,
    NanosetDatasetsArgs,
    PretrainDatasetsArgs,
)
from nanotron.helpers import (
    compute_remain_train_steps_of_a_data_stage_from_ckp,
    get_consumed_train_samples_of_a_data_stage_from_ckp,
)
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from nanotron.utils import main_rank_first

try:
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer
    from transformers import __version__ as tf_version
except ImportError:
    hf_hub_version = None
    tf_version = None

from nanotron.data.collator import MultilingualNanosetDataCollatorForCLM, NanosetDataCollatorForCLM

logger = logging.get_logger(__name__)


# Adapted from h4/src/h4/data/loading.py
def get_datasets(
    hf_dataset_or_datasets: Union[dict, str],
    hf_dataset_config_name: str,
    splits: Optional[Union[List[str], str]] = ["train", "test"],
) -> "DatasetDict":
    """
    Function to load dataset directly from DataArguments.

    Args:
        hf_dataset_or_datasets (Union[dict, str]): dict or string. When all probabilities are 1, we concatenate the datasets instead of sampling from them.
        splits (Optional[List[str]], optional): Section of the dataset to load, defaults to "train", "test"
            Can be one of `train_ift`, `test_rl`, or `..._rm` etc. H4 datasets are divided into 6 subsets for training / testing.

    Returns
        DatasetDict: DatasetDict object containing the dataset of the appropriate section with test + train parts.
    """

    if isinstance(splits, str):
        splits = [splits]

    if isinstance(hf_dataset_or_datasets, dict):
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        raw_datasets = _get_dataset_mix(hf_dataset_or_datasets, splits=splits)
    elif isinstance(hf_dataset_or_datasets, str):
        # e.g. Dataset = "HuggingFaceH4/testing_alpaca_small"
        # Note this returns things other than just train/test, which may not be intended
        raw_datasets = DatasetDict()
        for split in splits:
            raw_datasets[split] = load_dataset(
                hf_dataset_or_datasets,
                hf_dataset_config_name,
                split=split,
            )
    else:
        raise ValueError(f"hf_dataset_or_datasets must be a dict or string but is {type(hf_dataset_or_datasets)}")

    return raw_datasets


# Adapted from h4/src/h4/data/loading.py
def _get_dataset_mix(dataset_dict: dict, splits: List[str] = None, seed=42) -> "DatasetDict":
    """
    Helper function to load dataset mix from dict configuration.

    Args:
        dataset_dict (dict): Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], optional): Section of the dataset to load, defaults to "train", "test"
            Can be one of `train_{ift,rm,rl}` and `test_{ift,rm,rl}`. Our datasets are typically divided into 6 subsets for training / testing.
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_test_datasets = []
    fracs = []
    for ds, frac in dataset_dict.items():
        if frac < 0:
            raise ValueError(f"Dataset fraction for dataset {ds} is negative. (= {frac})")

        fracs.append(frac)
        for split in splits:
            if "train" in split:
                raw_train_datasets.append(
                    load_dataset(
                        ds,
                        split=split,
                    )
                )
            elif "test" in split:
                raw_test_datasets.append(
                    load_dataset(
                        ds,
                        split=split,
                    )
                )
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=seed)

    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_test_datasets) > 0:
        raw_datasets["test"] = concatenate_datasets(raw_test_datasets).shuffle(seed=seed)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_dict} not recognized with split {split}. Check the dataset has been correctly formatted."
        )

    return raw_datasets


def dummy_infinite_data_generator(
    micro_batch_size: int,
    sequence_length: int,
    input_pp_rank: int,
    output_pp_rank: int,
    vocab_size: int,
    seed: int,
    parallel_context: ParallelContext,
):
    def data_generator() -> Generator[Dict[str, Union[torch.Tensor, TensorPointer]], None, None]:
        # Random generator
        generator = torch.Generator(device="cuda")
        # Make sure that TP are synced always
        generator.manual_seed(
            seed * (1 + dist.get_rank(parallel_context.dp_pg)) * (1 + dist.get_rank(parallel_context.pp_pg))
        )

        while True:
            yield {
                "input_ids": torch.randint(
                    0,
                    vocab_size,
                    (micro_batch_size, sequence_length),
                    dtype=torch.long,
                    device="cuda",
                    generator=generator,
                )
                if dist.get_rank(parallel_context.pp_pg) == input_pp_rank
                else TensorPointer(group_rank=input_pp_rank),
                "input_mask": torch.ones(
                    micro_batch_size,
                    sequence_length,
                    dtype=torch.bool,
                    device="cuda",
                )
                if dist.get_rank(parallel_context.pp_pg) == input_pp_rank
                else TensorPointer(group_rank=input_pp_rank),
                "label_ids": torch.randint(
                    0,
                    vocab_size,
                    (micro_batch_size, sequence_length),
                    dtype=torch.long,
                    device="cuda",
                    generator=generator,
                )
                if dist.get_rank(parallel_context.pp_pg) == output_pp_rank
                else TensorPointer(group_rank=output_pp_rank),
                "label_mask": torch.ones(
                    micro_batch_size,
                    sequence_length,
                    dtype=torch.bool,
                    device="cuda",
                )
                if dist.get_rank(parallel_context.pp_pg) == output_pp_rank
                else TensorPointer(group_rank=output_pp_rank),
            }

    return data_generator


# Adapted from https://github.com/huggingface/accelerate/blob/a73898027a211c3f6dc4460351b0ec246aa824aa/src/accelerate/data_loader.py#L781C1-L824C28
class SkipBatchSampler(BatchSampler):
    """
    A `torch.utils.data.BatchSampler` that skips the first `n` batches of another `torch.utils.data.BatchSampler`.
    Note that in case of DDP, we skip batches on each rank, so a total of `skip_batches * parallel_context.dp_pg.size()` batches
    """

    def __init__(self, batch_sampler: BatchSampler, skip_batches: int, dp_size: int):
        self.batch_sampler = batch_sampler
        # In case of DDP, we skip batches on each rank, so a total of `skip_batches * parallel_context.dp_pg.size()` batches
        self.skip_batches = skip_batches // dp_size

    def __iter__(self):
        for index, samples in enumerate(self.batch_sampler):
            if index >= self.skip_batches:
                yield samples

    @property
    def total_length(self):
        return len(self.batch_sampler)

    def __len__(self):
        return len(self.batch_sampler) - self.skip_batches


def set_tensor_pointers(
    input_dict: Dict[str, Union[torch.Tensor, TensorPointer]], group: dist.ProcessGroup, group_rank: int
) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
    """Make sure only the group_rank rank has the data, others have TensorPointers."""
    return {
        k: v if dist.get_rank(group) == group_rank else TensorPointer(group_rank=group_rank)
        for k, v in input_dict.items()
    }


### CAUSAL LANGUAGE MODELING ###
def clm_process(
    raw_dataset: "Dataset",
    tokenizer: "PreTrainedTokenizerBase",
    text_column_name: str,
    dataset_processing_num_proc_per_process: int,
    dataset_overwrite_cache: bool,
    sequence_length: int,
):
    """Concatenate all texts from raw_dataset and generate chunks of `sequence_length + 1`, where chunks overlap by a single token."""
    # Adapted from https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/examples/pytorch/language-modeling/run_clm.py#L391-L439

    def group_texts(examples: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        # Concatenate all texts.
        concatenated_examples = {k: np.concatenate(v) for k, v in examples.items()}
        total_length = len(concatenated_examples[next(iter(examples.keys()))])
        # WARNING: We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= sequence_length + 1:
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1
        # Split by chunks of sequence_length.
        result = {
            k: [
                t[i : i + sequence_length + 1] for i in range(0, total_length - (sequence_length + 1), sequence_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    def _tokenize_and_group_texts(texts: List[str]) -> Dict[str, List[np.ndarray]]:
        tokenized_batch = tokenizer.batch_encode_plus(texts, return_attention_mask=False, return_token_type_ids=False)
        tokenized_batch = {k: [np.array(tokenized_texts) for tokenized_texts in v] for k, v in tokenized_batch.items()}
        return group_texts(tokenized_batch)

    train_dataset = raw_dataset.map(
        _tokenize_and_group_texts,
        input_columns=text_column_name,
        remove_columns=raw_dataset.column_names,
        features=Features({"input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1)}),
        batched=True,
        num_proc=dataset_processing_num_proc_per_process,
        load_from_cache_file=not dataset_overwrite_cache,
        desc=f"Grouping texts in chunks of {sequence_length+1}",
    )
    return train_dataset


# Adapted from: https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/data/data_collator.py#L607
@dataclasses.dataclass
class DataCollatorForCLM:
    """
    Data collator used for causal language modeling.

    - input_pp_rank: Discards last input id token
    - output_pp_rank: Discards first label id token
    - other pp ranks: Don't have data. Instead, we use `TensorPointer` to point to the rank having the data.
    """

    sequence_length: int
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext

    def __call__(self, examples: List[Dict[str, List[np.ndarray]]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Process the case when current rank doesn't require data. We return `TensorPointer` that points to ranks having the data.
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        if current_pp_rank not in [
            self.input_pp_rank,
            self.output_pp_rank,
        ]:
            assert all(len(example) == 0 for example in examples)
            return {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "input_mask": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
            }

        # Make sure we load only what's necessary, ie we only load a `input_ids` column.
        assert all(list(example.keys()) == ["input_ids"] for example in examples)

        # TODO @nouamanetazi: Is it better to have examples as np.array or torch.Tensor?
        input_ids = np.vstack([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s)
        batch_size, expanded_input_length = input_ids.shape

        result: Dict[str, Union[np.ndarray, TensorPointer]] = {}

        result["input_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["input_mask"] = TensorPointer(group_rank=self.input_pp_rank)
        result["label_ids"] = TensorPointer(group_rank=self.output_pp_rank)
        result["label_mask"] = TensorPointer(group_rank=self.output_pp_rank)

        assert (
            expanded_input_length == self.sequence_length + 1
        ), f"Samples should be of length {self.sequence_length + 1} (seq_len+1), but got {expanded_input_length}"

        # Process inputs: last token is the label
        if current_pp_rank == self.input_pp_rank:
            result["input_ids"] = input_ids[:, :-1]
            result["input_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

        # Process labels: shift them to the left
        if current_pp_rank == self.output_pp_rank:
            result["label_ids"] = input_ids[:, 1:]
            result["label_mask"] = np.ones((batch_size, self.sequence_length), dtype=np.bool_)

        if isinstance(result["input_ids"], torch.Tensor) and result["input_ids"].shape[-1] != self.sequence_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['input_ids'].shape[-1]}, but should be"
                f" {self.sequence_length}."
            )
        if isinstance(result["label_ids"], torch.Tensor) and result["label_ids"].shape[-1] != self.sequence_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['label_ids'].shape[-1]}, but should be"
                f" {self.sequence_length}."
            )

        # Cast np.array to torch.Tensor
        result = {k: v if isinstance(v, TensorPointer) else torch.from_numpy(v) for k, v in result.items()}
        return result


# Adapted from https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L763-L835
def get_sampler(
    dl_ranks_size: int,
    dl_rank: int,
    train_dataset: Union["Dataset", torch.utils.data.Dataset],
    consumed_train_samples: int,
    seed: int = 42,
    use_loop_to_round_batch_size: bool = False,
    micro_batch_size: Optional[int] = None,
    drop_last: Optional[bool] = True,
    shuffle: bool = True,
) -> Optional[torch.utils.data.Sampler]:
    """returns sampler that restricts data loading to a subset of the dataset proper to the DP rank"""

    # Build the sampler.
    # TODO @nouamanetazi: Support group_by_length: https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L783-L810

    if use_loop_to_round_batch_size:
        assert micro_batch_size is not None
        # loops at the end back to the beginning of the shuffled samples to make each process have a round multiple of batch_size samples.
        sampler = DistributedSamplerWithLoop(
            train_dataset,
            batch_size=micro_batch_size,
            num_replicas=dl_ranks_size,
            rank=dl_rank,
            seed=seed,
            drop_last=drop_last,
        )
    else:
        sampler = DistributedSampler(
            train_dataset, num_replicas=dl_ranks_size, rank=dl_rank, seed=seed, drop_last=drop_last, shuffle=shuffle
        )

    if consumed_train_samples > 0:
        sampler = SkipBatchSampler(sampler, skip_batches=consumed_train_samples, dp_size=dl_ranks_size)

    return sampler


# Adapted from https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L837
def _get_train_hf_dataloader(
    train_dataset: "Dataset",
    sequence_length: int,
    parallel_context: ParallelContext,
    input_pp_rank: int,
    output_pp_rank: int,
    micro_batch_size: int,
    consumed_train_samples: int,
    dataloader_num_workers: int,
    seed_worker: int,
    dataloader_drop_last: bool = True,
    dataloader_pin_memory: bool = True,
    use_loop_to_round_batch_size: bool = False,
) -> DataLoader:
    if not isinstance(train_dataset, datasets.Dataset):
        raise ValueError(f"training requires a datasets.Dataset, but got {type(train_dataset)}")

    # Case of ranks requiring data
    if dist.get_rank(parallel_context.pp_pg) in [
        input_pp_rank,
        output_pp_rank,
    ]:
        train_dataset = train_dataset.with_format(type="numpy", columns=["input_ids"], output_all_columns=True)

    # Case of ranks not requiring data. We give them an infinite dummy dataloader
    else:
        #
        assert train_dataset.column_names == ["input_ids"], (
            f"Dataset has to have a single column, with `input_ids` as the column name. "
            f"Current dataset: {train_dataset}"
        )
        dataset_length = len(train_dataset)
        train_dataset = train_dataset.remove_columns(column_names="input_ids")
        assert (
            len(train_dataset) == 0
        ), f"Dataset has to be empty after removing the `input_ids` column. Current dataset: {train_dataset}"
        # HACK as if we remove the last column of a train_dataset, it becomes empty and it's number of rows becomes empty.
        train_dataset = EmptyInfiniteDataset(length=dataset_length)
        # No need to spawn a lot of workers, we can just use main
        dataloader_num_workers = 0

    data_collator = DataCollatorForCLM(
        sequence_length=sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        parallel_context=parallel_context,
    )

    # Compute size and rank of dataloader workers
    dp_ranks_size = parallel_context.dp_pg.size()
    dp_rank = parallel_context.dp_pg.rank()

    # TODO @nouamanetazi: Remove unused columns: https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L852
    # TODO @nouamanetazi: Support torch.utils.data.IterableDataset: https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/src/transformers/trainer.py#L855-L872

    train_sampler = get_sampler(
        dl_rank=dp_rank,
        dl_ranks_size=dp_ranks_size,
        train_dataset=train_dataset,
        seed=seed_worker,
        use_loop_to_round_batch_size=use_loop_to_round_batch_size,
        micro_batch_size=micro_batch_size,
        drop_last=dataloader_drop_last,
        consumed_train_samples=consumed_train_samples,
    )

    return DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        drop_last=dataloader_drop_last,  # we also drop_last in `clm_process()`
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dp_rank),
        # TODO @thomasw21: I'm not sure but this doesn't seem to work at all.
        # pin_memory_device="cuda",
    )

def get_dataloader_from_data_stage(
    trainer: DistributedTrainer,
    data: DataArgs,
    consumed_train_samples: int,
    num_remaining_train_steps: int,
):
    """
    Returns a dataloader for a given data stage.

    data: The data configuration for the current stage.
    consumed_train_samples: The number of samples consumed by the model in the this stage (each stage starts from zero).
    num_remaining_train_steps: The number of remaining training steps for this stage.
    """
    assert consumed_train_samples >= 0, "consumed_train_samples should be greater than 0"
    assert num_remaining_train_steps >= 0, "num_remaining_train_steps should be greater than 0"

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    # Case 1: Dummy data generator
    if data.dataset is None:
        log_rank("Using dummy data generator", logger=logger, level=logging.INFO, rank=0)
        dataloader = dummy_infinite_data_generator(
            micro_batch_size=trainer.micro_batch_size,
            sequence_length=trainer.sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            vocab_size=trainer.model_config.vocab_size,
            seed=data.seed,
            parallel_context=trainer.parallel_context,
        )()

    # Case 2: HuggingFace datasets
    elif isinstance(data.dataset, PretrainDatasetsArgs):
        log_rank("Using `datasets` library", logger=logger, level=logging.INFO, rank=0)
        tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
        log_rank(
            f"Loading tokenizer from {tokenizer_path} and transformers/hf_hub versions {tf_version, hf_hub_version}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        # We need to the 1st device to process dataset and cache it, then other devices load from cache
        with main_rank_first(trainer.parallel_context.world_pg):
            # TODO @nouamanetazi: this may timeout before 1st device finishes processing dataset. Can we have a ctxmanager to modify timeout?
            # TODO: generalise to include  for validation/test splits

            # We load the raw dataset
            raw_dataset = get_datasets(
                hf_dataset_or_datasets=data.dataset.hf_dataset_or_datasets,
                hf_dataset_config_name=data.dataset.hf_dataset_config_name,
                splits=data.dataset.hf_dataset_splits,
            )["train"]

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            # Check that tokenizer's vocab size is smaller than the model's vocab size
            assert (
                tokenizer.vocab_size <= trainer.model_config.vocab_size
            ), f"Tokenizer's vocab size ({tokenizer.vocab_size}) is larger than the model's vocab size ({trainer.model_config.vocab_size})"

            # We apply the Causal Language Modeling preprocessing
            train_dataset = clm_process(
                raw_dataset=raw_dataset,
                tokenizer=tokenizer,
                text_column_name=data.dataset.text_column_name,
                dataset_processing_num_proc_per_process=data.dataset.dataset_processing_num_proc_per_process,
                dataset_overwrite_cache=data.dataset.dataset_overwrite_cache,
                sequence_length=trainer.sequence_length,
            )

            # We load the processed dataset on the ranks requiring it
            dataloader = _get_train_hf_dataloader(
                train_dataset=train_dataset,
                sequence_length=trainer.sequence_length,
                parallel_context=trainer.parallel_context,
                input_pp_rank=input_pp_rank,
                output_pp_rank=output_pp_rank,
                micro_batch_size=trainer.micro_batch_size,
                consumed_train_samples=consumed_train_samples,
                dataloader_num_workers=data.num_loading_workers,
                seed_worker=data.seed,
                dataloader_drop_last=True,
            )

            # Check if we have enough samples for train_steps
            total_tokens_dataset = len(dataloader.dataset) * trainer.sequence_length
            num_tokens_needed_for_training = (
                num_remaining_train_steps * trainer.global_batch_size * trainer.sequence_length
            )
            assert num_tokens_needed_for_training <= total_tokens_dataset, (
                f"Dataset is too small for steps ({total_tokens_dataset} < {num_tokens_needed_for_training}), "
                f"Try train_steps<={len(dataloader.dataset) // trainer.global_batch_size + trainer.iteration_step}"
            )

    # Case 3: Nanosets
    elif isinstance(data.dataset, NanosetDatasetsArgs):
        # Get tokenizer cardinality
        tokenizer = AutoTokenizer.from_pretrained(trainer.config.tokenizer.tokenizer_name_or_path)
        token_size = 4 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else 2
        del tokenizer
        # Create Nanoset
        from nanotron.data.nanoset import Nanoset

        with main_rank_first(trainer.parallel_context.world_pg):
            train_dataset = Nanoset(
                dataset_folders=data.dataset.dataset_folder,
                dataset_weights=data.dataset.dataset_weights,
                sequence_length=trainer.sequence_length,
                token_size=token_size,
                train_split_num_samples=trainer.config.tokens.train_steps * trainer.global_batch_size,
                random_seed=data.seed,
            )

        # Prepare dataloader
        train_dataloader = build_nanoset_dataloader(
            train_dataset,
            trainer.sequence_length,
            parallel_context=trainer.parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=trainer.micro_batch_size,
            consumed_train_samples=consumed_train_samples,
            dataloader_num_workers=data.num_loading_workers,
            dataloader_drop_last=True,
        )

        return train_dataloader
    # Case 4: MultilingualNanosets
    elif isinstance(data.dataset, MultilingualNanosetDatasetsArgs):
        # Get tokenizer cardinality
        tokenizer = AutoTokenizer.from_pretrained(trainer.config.tokenizer.tokenizer_name_or_path)
        token_size = 4 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else 2
        del tokenizer
        # Create Nanoset
        from nanotron.data.multilingual_nanoset import MultilingualNanoset

        with main_rank_first(trainer.parallel_context.world_pg):
            train_dataset = MultilingualNanoset(
                dataset_folders=data.dataset.training_folder,
                dataset_weights=data.dataset.dataset_weights,
                sequence_length=trainer.sequence_length,
                token_size=token_size,
                train_split_num_samples=trainer.config.tokens.train_steps * trainer.global_batch_size,
                random_seed=data.seed,
            )

        # Prepare dataloader
        train_dataloader = build_nanoset_dataloader(
            train_dataset,
            trainer.sequence_length,
            parallel_context=trainer.parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=trainer.micro_batch_size,
            consumed_train_samples=consumed_train_samples,
            dataloader_num_workers=data.num_loading_workers,
            dataloader_drop_last=True,
            is_multilingual=True,
        )

        return train_dataloader
    else:
        raise ValueError(f"Unhandled case of `self.config.data.dataset`. Got: {data.dataset}")

    return dataloader


def get_valid_dataloader_from_data_stage(
    trainer: DistributedTrainer,
    data: DataArgs,
    # consumed_train_samples: int, We will never use this because in each valid iteration we consume all the samples
):

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    # Only support Validation with MultilingualNanosets
    if isinstance(data.dataset, MultilingualNanosetDatasetsArgs):
        # Get tokenizer cardinality
        tokenizer = AutoTokenizer.from_pretrained(trainer.config.tokenizer.tokenizer_name_or_path)
        token_size = 4 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else 2
        del tokenizer
        # Create Multilingual Nanoset
        from nanotron.data.multilingual_nanoset import MultilingualNanoset

        with main_rank_first(trainer.parallel_context.world_pg):
            valid_dataset = MultilingualNanoset(
                dataset_folders=data.dataset.validation_folder,
                sequence_length=trainer.sequence_length,
                token_size=token_size,
                is_valid=True,
                random_seed=data.seed,
            )

        # Prepare dataloader
        valid_dataloader = build_nanoset_dataloader(
            valid_dataset,
            trainer.sequence_length,
            parallel_context=trainer.parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=trainer.micro_batch_size,
            dataloader_num_workers=data.num_loading_workers,
            dataloader_drop_last=True,
            shuffle=True,
            is_multilingual=True,
        )

        return valid_dataloader
    else:
        raise ValueError(
            f"Unhandled case of `self.config.data.dataset`. Got: {data.dataset}. Validation is currently just supported for MultilingualNanoset"
        )


def get_train_dataloader(trainer: DistributedTrainer) -> Dict[str, DataLoader]:
    dataloaders = {}

    for stage_idx, stage in enumerate(trainer.config.data_stages):
        # NOTE: we only create the dataloader for the first stage,
        # then we lazy initialize the dataloader for the other stages
        stage = cast(DatasetStageArgs, stage)
        consumed_train_samples = get_consumed_train_samples_of_a_data_stage_from_ckp(stage, trainer.metadata)
        assert (
            consumed_train_samples is not None
        ), f"Cannot find consumed_train_samples for stage {stage.start_training_step} in the checkpoint"

        num_remaining_train_steps = compute_remain_train_steps_of_a_data_stage_from_ckp(
            stage, trainer.config, trainer.metadata
        )
        log_rank(
            f"[Training Plan] Stage {stage.name} has {num_remaining_train_steps} remaining training steps and has consumed {consumed_train_samples} samples",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        dataloader = (
            get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                num_remaining_train_steps=num_remaining_train_steps,
            )
            if stage_idx == 0
            else lambda stage=stage: get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                num_remaining_train_steps=num_remaining_train_steps,
            )
        )
        dataloaders[stage.name] = dataloader
    return dataloaders


def get_valid_dataloader(trainer: DistributedTrainer) -> Dict[str, DataLoader]:
    dataloaders = {}

    for stage_idx, stage in enumerate(trainer.config.data_stages):
        # NOTE: we only create the dataloader for the first stage,
        # then we lazy initialize the dataloader for the other stages
        stage = cast(DatasetStageArgs, stage)

        log_rank(
            f"[Validation Plan] Stage {stage.name} has {len(stage.data.dataset.validation_folder)} folders with samples for the validation set",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        dataloader = (
            get_valid_dataloader_from_data_stage(trainer, stage.data)
            if stage_idx == 0
            else lambda stage=stage: get_valid_dataloader_from_data_stage(trainer, stage.data)
        )
        # TODO(tj.solergibert) As we are creating again the valid dataloader in every validation stage, we print multiple times
        # the validation MultilingualNanoset info (Number of samples, etc.) [UPDATE: ]. In order to solve that, we could get rid of this lambda
        # funcs and directly create all dataloaders.
        #
        # This lambda functs (Used in training too) are for creating the DataLoaders lazyly FOR 1. Start training faster instead
        # of creating multiple DataLoaders 2. Consume less memory as the lambda func is lighter that the DataLoader object with
        # the Dataset, collator, etc.
        # BUT 1. The Nanoset creation process is very fast and 2. Nanosets doesn't consume any memory at all till we start sampling
        # from the Nanoset. Also they later transform the DataLoader into a Iterator object so it's impossible to retrieve
        # the DataLoader object again to delete it (More comments in trainer.py)
        dataloaders[stage.name] = dataloader
    return dataloaders


def get_dataloader_worker_init(dp_rank: int):
    """Creates random states for each worker in order to get different state in each workers"""

    def dataloader_worker_init(worker_id):
        # Dataloader is TP/PP synced in random states
        seed = 2 ** (1 + worker_id) * 3 ** (1 + dp_rank) % (2**32)
        set_random_seed(seed)

    return dataloader_worker_init

def build_nanoset_dataloader(
    dataset,
    sequence_length: int,
    parallel_context: ParallelContext,
    input_pp_rank: int,
    output_pp_rank: int,
    micro_batch_size: int,
    dataloader_num_workers: int,
    is_multilingual: bool = False,
    consumed_train_samples: int = 0,
    dataloader_drop_last: bool = True,
    dataloader_pin_memory: bool = True,
    shuffle: bool = False,
) -> DataLoader:

    # Case of ranks not requiring data. We give them a dummy dataset, then the collator will do his job
    if dist.get_rank(parallel_context.pp_pg) not in [input_pp_rank, output_pp_rank]:
        dataset_length = len(dataset)
        dataset = EmptyInfiniteDataset(length=dataset_length)
        # No need to spawn a lot of workers, we can just use main
        dataloader_num_workers = 0

    data_collator = NanosetDataCollatorForCLM(
        sequence_length=sequence_length,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        parallel_context=parallel_context,
    )

    if is_multilingual:
        data_collator = MultilingualNanosetDataCollatorForCLM(
            sequence_length=sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            parallel_context=parallel_context,
        )

    # Compute size and rank of dataloader workers
    dp_ranks_size = parallel_context.dp_pg.size()
    dp_rank = parallel_context.dp_pg.rank()

    sampler = get_sampler(
        train_dataset=dataset,
        dl_ranks_size=dp_ranks_size,
        dl_rank=dp_rank,
        drop_last=dataloader_drop_last,
        consumed_train_samples=consumed_train_samples,
        shuffle=shuffle,
    )

    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        collate_fn=data_collator,
        drop_last=dataloader_drop_last,
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        worker_init_fn=get_dataloader_worker_init(dp_rank=dp_rank),
    )

class EmptyInfiniteDataset:
    """Hack as removing all columns from a datasets.Dataset makes the number of rows 0."""

    def __init__(self, length: int):
        self._length = length

    def __getitem__(self, item) -> Dict:
        if isinstance(item, int):
            return {}
        raise NotImplementedError(f"{item} of type {type(item)} is not supported yet")

    def __len__(self) -> int:
        return self._length
