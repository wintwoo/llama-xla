import inspect
import logging
import os
import torch
from torch import nn
from torch_xla.distributed.fsdp import (
    XlaFullyShardedDataParallel as FSDP,
    checkpoint_module,
)
from transformers.models.llama import (
    LlamaConfig,
    LlamaModel,
    LlamaModel,
    LlamaForCausalLM,
    LlamaPreTrainedModel,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _xla_fsdp_wrap(module, use_grad_checkpoint=True):
    if use_grad_checkpoint:
        module = checkpoint_module(module)
    module = FSDP(module, compute_dtype=torch.bfloat16, shard_param_on_dim_0=True, pin_layout_in_collective_ops=True)
    return module

def _set_block_weights_from_checkpoint(module, block_num, presharded_checkpoints):
    with torch.no_grad():
        ckpt = torch.load(os.path.join(presharded_checkpoints, f"model.layers.{block_num}.bin"))
        # this is a module buffer, not a param
        module.self_attn.rotary_emb.inv_freq = ckpt.pop("self_attn.rotary_emb.inv_freq").type(torch.float32)
        module.load_state_dict(ckpt)
    return module

def get_wrapped_llama_from_config(config, presharded_checkpoints: str=None, use_grad_checkpoint=True):
    model = LlamaXlaFsdpForCausalLM(config, presharded_checkpoints)
    # wrap model at root
    forward_signature = inspect.signature(model.forward.__func__)
    model = _xla_fsdp_wrap(model, use_grad_checkpoint=use_grad_checkpoint)
    model.forward.__func__.__signature__ = forward_signature
    return model

class LlamaXlaFsdpModel(LlamaModel):

    def __init__(self, config: LlamaConfig, presharded_checkpoints: str=None):
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if presharded_checkpoints is None:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        else:
            logger.debug("Using token embeddings from checkpoint")
            ckpt = torch.load(os.path.join(presharded_checkpoints, "model.embed_tokens.weight.bin"))
            self.embed_tokens = nn.Embedding.from_pretrained(ckpt["model.embed_tokens.weight"].type(torch.float32))

        blocks = []
        for i in range(config.num_hidden_layers):
            block = LlamaDecoderLayer(config)
            if presharded_checkpoints is None:
                block.apply(self._init_weights) 
            else:
                logger.debug(f"Using checkpoint weights for decoder block {i}")
                block = _set_block_weights_from_checkpoint(block, i, presharded_checkpoints)
                
            block = _xla_fsdp_wrap(block, use_grad_checkpoint=True)
            blocks.append(block)

        self.layers = nn.ModuleList(blocks)

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if presharded_checkpoints is None:
            self.norm.apply(self._init_weights)
        else:
            with torch.no_grad():
                logger.debug("Using checkpoint weights for RMSNorm")
                ckpt = torch.load(os.path.join(presharded_checkpoints, "model.norm.weight.bin"))
                self.norm.weight = nn.Parameter(ckpt["model.norm.weight"].type(torch.float32))

        self.gradient_checkpointing = False

        # initialize weights and apply final processing
        self.post_init()
    
    def post_init(self):
        self.tie_weights()
        self._backward_compatibility_gradient_checkpointing()


class LlamaXlaFsdpForCausalLM(LlamaForCausalLM):

    def __init__(self, config, presharded_checkpoints: str=None):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlamaXlaFsdpModel(config, presharded_checkpoints)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if presharded_checkpoints is None:
            with torch.no_grad():
                logger.debug("Using checkpoint weights for LM head")
                ckpt = torch.load(os.path.join(presharded_checkpoints, "lm_head.weight.bin"))
                self.lm_head.weight = nn.Parameter(ckpt["lm_head.weight"].type(torch.float32))

        # initialize weights and apply final processing
        self.post_init()

    def post_init(self):
        self.tie_weights()
        self._backward_compatibility_gradient_checkpointing()
