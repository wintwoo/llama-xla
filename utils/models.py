import logging
from models.llama import (
    LlamaXlaFsdpForCausalLM,
    get_wrapped_llama_from_config,
)
import os
from torch.optim import Optimizer
import torch_xla.core.xla_model as xm
from torch_xla.amp.syncfree import AdamW
from transformers import LlamaConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(module)s:%(funcName)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_model(
        config_file_path: str,
        resharded_checkpoint_dir: str=None,
        use_grad_checkpoint: bool=False,
):
    logger.info("Loading model")
    config = LlamaConfig.from_pretrained(config_file_path)
    model = get_wrapped_llama_from_config(
        config=config,
        use_grad_checkpoint=use_grad_checkpoint,
        resharded_checkpoint_dir=resharded_checkpoint_dir,
    )
    model.train()
    return model

def checkpoint_model(
        model: LlamaXlaFsdpForCausalLM,
        optimizer: Optimizer,
        output_dir: str,
        ckpt_name: str,
):
    ckpt_dir = os.path.join(output_dir, ckpt_name)
    ckpt_file_path = os.path.join(
        ckpt_dir,
        f"ckpt_rank-{xm.get_ordinal():08d}-of-{xm.xrt_world_size():08d}.pth"
    )
    ckpt = {
        'model': model.state_dict(),
        'shard_metadata': model.get_shard_metadata(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,  # not needed in ckpt consolidation
    }
    os.makedirs(ckpt_dir, exist_ok=True)
    xm.save(ckpt, ckpt_file_path, master_only=False)
    logger.info(f"Checkpoint saved to {ckpt_dir}")

def save_model(
        model: LlamaXlaFsdpForCausalLM,
        output_dir: str,
        consolidate_checkpoint: bool = False,
):
    checkpoint_model(model, None, output_dir, "consolidated_model")
    if consolidate_checkpoint:
        logger.info("Waiting for all ranks")
        xm.rendezvous("consolidate_model")
        if xm.is_master_ordinal(local=False):
            logger.info("Saving consolidated model")
            from torch_xla.distributed.fsdp import consolidate_sharded_model_checkpoints
            model_dir = os.path.join(output_dir, "final_model")
            os.makedirs(model_dir, exist_ok=True)
            consolidate_sharded_model_checkpoints(
                ckpt_prefix=os.path.join(model_dir, "ckpt_rank"),
                ckpt_suffix="-*-of-*.pth",
            )
            logger.info(f"Consolidated model saved to {model_dir}")