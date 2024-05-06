from typing import Any
from transformer_lens import HookedTransformer


def make_iit_hl_model(hl_model):
    class IITHLModel:
        """A wrapper class to make tracr models compatible with IITModelPair"""
        def __init__(self, hl_model: HookedTransformer):
            self.hl_model = hl_model

        def __getattr__(self, name: str):
            if hasattr(self.hl_model, name):
                return getattr(self.hl_model, name)
            else:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        def forward(self, input):
            x, _, _ = input
            out = self.hl_model(x)
            if self.hl_model.is_categorical():
                return out.argmax(dim=1)
            return out

        def run_with_hooks(self, input, *args, **kwargs):
            x, _, _ = input
            out = self.hl_model.run_with_hooks(x, *args, **kwargs)
            if self.hl_model.is_categorical():
                return out.argmax(dim=1)
            return out

        def run_with_cache(self, input):
            x, _, _ = input
            out, cache = self.hl_model.run_with_cache(x)
            if self.hl_model.is_categorical():
                return out.argmax(dim=1), cache
            return out, cache
        
        def __call__(self, *args: Any, **kwds: Any) -> Any:
            return self.forward(*args, **kwds)

    return IITHLModel(hl_model)
