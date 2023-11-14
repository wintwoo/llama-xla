from itertools import chain
from . import (
    samsum,
    wikitext,
)

DATASET_PREPROC = {
    "samsum": samsum.preprocess,
    "wikitext": wikitext.preprocess,
}

def get_preprocessed_dataset(preproc_routine):
    if preproc_routine is not None and preproc_routine not in DATASET_PREPROC.keys():
        raise NotImplementedError(f"The `{preproc_routine}` preprocessing function is not implemented!")
    return DATASET_PREPROC[preproc_routine]

def group_texts(examples, block_size=1024):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result