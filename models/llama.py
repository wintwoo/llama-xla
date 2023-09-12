import inspect
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


def _xla_fsdp_wrap(module, use_grad_checkpoint=True):
    if use_grad_checkpoint:
        module = checkpoint_module(module)
    module = FSDP(module, compute_dtype=torch.bfloat16, shard_param_on_dim_0=True, pin_layout_in_collective_ops=True)
    return module


def get_wrapped_llama_from_config(config, use_grad_checkpoint=True):
    model = LlamaXlaFsdpForCausalLM(config)
    # Wrap model at root
    forward_signature = inspect.signature(model.forward.__func__)
    model = _xla_fsdp_wrap(model, use_grad_checkpoint=use_grad_checkpoint)
    model.forward.__func__.__signature__ = forward_signature
    return model


class LlamaXlaFsdpModel(LlamaModel):

    def __init__(self, config: LlamaConfig):
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        blocks = []
        for i in range(config.num_hidden_layers):
            block = LlamaDecoderLayer(config)
            block.apply(self._init_weights)
            block = _xla_fsdp_wrap(block, use_grad_checkpoint=True)
            blocks.append(block)

        self.layers = nn.ModuleList(blocks)

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm.apply(self._init_weights)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    
    def post_init(self):
        self.tie_weights()
        self._backward_compatibility_gradient_checkpointing()


class LlamaXlaFsdpForCausalLM(LlamaForCausalLM):

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlamaXlaFsdpModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def post_init(self):
        self.tie_weights()
        self._backward_compatibility_gradient_checkpointing()
