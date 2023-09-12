import transformers
from transformers.testing_utils import CaptureLogger

tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")


def preprocess(examples, tokenizer):
    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n{{summary}}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return prompt.format(
            dialog=sample["dialogue"],
            summary=sample["summary"],
            eos_token=tokenizer.eos_token,
        )

    d_s = zip(examples["dialogue"], examples["summary"])
    processed = [apply_prompt_template({"dialogue": x[0], "summary": x[1]}) for x in d_s]

    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(processed)
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
    return output
