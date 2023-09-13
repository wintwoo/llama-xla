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


def load_model(config_file_path: str, presharded_checkpoints: str=None):
    logger.info("Loading model")
    config = LlamaConfig.from_pretrained(config_file_path)
    model = get_wrapped_llama_from_config(
        config, presharded_checkpoints=presharded_checkpoints
    )
    model.train()
    return model

def save_model(model: LlamaXlaFsdpForCausalLM, optimizer: Optimizer, output_dir: str):
    ckpt_path = os.path.join(
        output_dir,
        f"final_ckpt_rank-{xm.get_ordinal():08d}-of-{xm.xrt_world_size():08d}.pth"
    )
    ckpt = {
        'model': model.state_dict(),
        'shard_metadata': model.get_shard_metadata(),
        'optimizer': optimizer.state_dict(),  # not needed in ckpt consolidation
    }
    os.makedirs(output_dir, exist_ok=True)
    xm.save(ckpt, ckpt_path, master_only=False)
    logging.info(f"checkpoint saved to {ckpt_path}")

    if xm.is_master_ordinal(local=False):
        consolidate_sharded_model_checkpoints(
            ckpt_prefix=os.path.join(output_dir, "final_ckpt"),
            ckpt_suffix="_rank-*-of-*.pth",
        )