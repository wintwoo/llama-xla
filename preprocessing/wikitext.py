import transformers
from transformers.testing_utils import CaptureLogger

tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")


def preprocess(examples, tokenizer):
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples)
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
    return output
