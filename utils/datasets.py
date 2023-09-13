import datasets
from preprocessing import utils as preproc_utils


def load_and_process_dataset(
    dataset: str,
    tokenizer,
    detect_columns_from_split: str = "train",
    block_size: int = 1024,
    num_cores: int = 8,
):
    # training dataset
    dataset = datasets.load_dataset(dataset)
    column_names = list(dataset[detect_columns_from_split].features)
    preproc_func = preproc_utils.get_preprocessed_dataset(dataset)
    tokenized_dataset = dataset.map(
        lambda x: preproc_func(x, tokenizer),
        batched=True,
        num_proc=num_cores,
        remove_columns=column_names,
    )
    packed_dataset = tokenized_dataset.map(
        lambda x: preproc_utils.group_texts(x, block_size=block_size),
        batched=True,
        num_proc=num_cores,
    )
    train_dataset = packed_dataset.with_format("torch")
    return train_dataset

