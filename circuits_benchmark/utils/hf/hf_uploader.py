from transformer_lens import HookedTransformerConfig, HookedTransformer
from transformers import PreTrainedModel, PretrainedConfig


class LLModelConfig(PretrainedConfig):
    def __init__(self, cfg: HookedTransformerConfig, 
                 task: str, 
                 tracr: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg.to_dict()
        self.task = task
        self.tracr = False

class TLModel(PreTrainedModel):
    def __init__(
        self,
        tl_model: HookedTransformer,
        config: LLModelConfig,
        *args,
        **kwargs
    ):
        super().__init__(config, *args, **kwargs)
        self.model = tl_model
        self.config = config
        self.model.to("cpu")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This model is not meant to be used for forward pass.")
