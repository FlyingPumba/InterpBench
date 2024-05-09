from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers import PreTrainedModel, AutoTokenizer, PretrainedConfig
from transformer_lens import HookedTransformerConfig
import torch
import enum

class output_type(enum.Enum):
    base_model_output = 1
    causal_lm_output = 2

class TLModel(PreTrainedModel):
    def __init__(
        self,
        tl_model: HookedTransformerConfig,
        config: PretrainedConfig,
        output_type: output_type = output_type.base_model_output,
        *inputs,
        **kwargs
    ):
        super().__init__(config, *inputs, **kwargs)
        self.model = tl_model
        self.model.to("cpu")
        self.tl_config = tl_model.cfg
        self.output_type = output_type

    def __getattr__(self, name):
        if name == "wte":
            w_E = self.model.embed.W_E.T
            return torch.nn.Embedding.from_pretrained(w_E.clone().detach())
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, *args, **kwargs):
        if len(args) > 0:
            x = args[0]
        else:
            x = kwargs.get("input_ids", None)
        if x is None:
            raise NotImplementedError("No input provided")
        out = self.model(x)
        if self.output_type == output_type.base_model_output:
            return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=out)
        elif self.output_type == output_type.causal_lm_output:
            return CausalLMOutputWithCrossAttentions(logits=out)
        else:
            raise NotImplementedError("Invalid output type")
