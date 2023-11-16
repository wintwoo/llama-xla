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

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_model(
        config_file_path: str,
        presharded_checkpoints: str=None,
        use_grad_checkpoint: bool=False,
):
    logger.info("Loading model")
    config = LlamaConfig.from_pretrained(config_file_path)
    model = get_wrapped_llama_from_config(
        config=config,
        use_grad_checkpoint=use_grad_checkpoint,
        presharded_checkpoints=presharded_checkpoints,
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
        'optimizer': optimizer.state_dict(),  # not needed in ckpt consolidation
    }
    os.makedirs(ckpt_dir, exist_ok=True)
    xm.save(ckpt, ckpt_file_path, master_only=False)
    logging.info(f"Checkpoint saved to {ckpt_dir}")

def save_model(model: LlamaXlaFsdpForCausalLM, optimizer: Optimizer, output_dir: str):
    checkpoint_model(model, optimizer, output_dir, "consolidated_model")
    if xm.is_master_ordinal(local=False):
        from torch_xla.distributed.fsdp import consolidate_sharded_model_checkpoints
        model_dir = os.path.join(output_dir, "consolidated_model")
        consolidate_sharded_model_checkpoints(
            ckpt_prefix=os.path.join(model_dir, "ckpt_rank"),
            ckpt_suffix="-*-of-*.pth",
        )
        logging.info(f"Consolidated model saved to {model_dir}")