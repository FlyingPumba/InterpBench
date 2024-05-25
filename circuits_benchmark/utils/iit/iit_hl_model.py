from typing import Any
from transformer_lens import HookedTransformer
import torch


def make_iit_hl_model(hl_model):
    class IITHLModel:
        """A wrapper class to make tracr models compatible with IITModelPair"""

        def __init__(self, hl_model: HookedTransformer):
            self.hl_model = hl_model
            self.hl_model.to(hl_model.device)
            for p in hl_model.parameters():
                p.requires_grad = False
                p.to(hl_model.device)

        def __getattr__(self, name: str):
            if hasattr(self.hl_model, name):
                return getattr(self.hl_model, name)
            else:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                )

        def create_hl_output(self, y):
            # if self.hl_model.is_categorical():
            #     argmax_y = y.argmax(dim=-1)
            #     # convert to one-hot
            #     return torch.nn.functional.one_hot(
            #         argmax_y, num_classes=y.shape[-1]
            #     )
            return y
        
        def get_correct_input(self, input):
            if isinstance(input, tuple):
                return input[0]
            elif isinstance(input, torch.Tensor):
                return input
            else:
                raise ValueError(f"Invalid input type: {type(input)}")
        
        def forward(self, input):
            x = self.get_correct_input(input)
            out = self.hl_model(x)
            return self.create_hl_output(out)

        def run_with_hooks(self, input, *args, **kwargs):
            x = self.get_correct_input(input)
            out = self.hl_model.run_with_hooks(x, *args, **kwargs)
            # if self.hl_model.is_categorical():
            #     return out.argmax(dim=1)
            return self.create_hl_output(out)

        def run_with_cache(self, input):
            x, _, _ = input
            out, cache = self.hl_model.run_with_cache(x)
            # if self.hl_model.is_categorical():
            #     return out.argmax(dim=1), cache
            return self.create_hl_output(out), cache

        def __call__(self, *args: Any, **kwds: Any) -> Any:
            return self.forward(*args, **kwds)

    return IITHLModel(hl_model)
