from interp_utils.lens import TunedLensConfig
from transformer_lens import HookedTransformer, ActivationCache
import torch
import torch.nn as nn
from fancy_einsum import einsum


class Translators(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def make_translators(cls, d_model, num_heads, num_layers, config):
        head_translators = {}
        mlp_translators = {}
        for layer in range(num_layers):
            for head in range(num_heads):
                head_translators[f"L{layer}H{head}"] = make_linear_map(d_model, config)
            mlp_translators[f"{layer}_mlp_out"] = make_linear_map(d_model, config)

        mlp_translators["embed"] = make_linear_map(d_model, config)
        mlp_translators["pos_embed"] = make_linear_map(d_model, config)

        return cls({**head_translators, **mlp_translators})

    def forward(self, cache: ActivationCache, pos_slice: slice):
        # get residual stack for each layer and head
        per_layer_residual, layers = cache.decompose_resid(
            -1, mode="mlp", return_labels=True, pos_slice=pos_slice
        )
        per_head_residual, attns = cache.stack_head_results(
            layer=-1, pos_slice=pos_slice, return_labels=True
        )
        preds = {}
        for per_layer_residual, layer in zip(per_layer_residual, layers):
            preds[layer] = self[layer]["model"](per_layer_residual.clone().detach())
        for per_head_residual, attn in zip(per_head_residual, attns):
            preds[attn] = self[attn]["model"](per_head_residual.clone().detach())
        return preds

    def iter_fn(self, fn):
        for translator in self.values():
            fn(translator)

    def model_iter_fn(self, fn):
        for translator in self.values():
            fn(translator["model"])

    def optimizer_iter_fn(self, fn):
        for translator in self.values():
            fn(translator["optimizer"])

    def zero_grad(self):
        self.optimizer_iter_fn(lambda opt: opt.zero_grad())

    def step(self):
        self.optimizer_iter_fn(lambda opt: opt.step())

    def train(self):
        self.model_iter_fn(lambda model: model.train())

    def eval(self):
        self.model_iter_fn(lambda model: model.eval())

    @staticmethod
    def apply_last_layer_to_translator_outputs(
        outputs: dict[str, torch.Tensor],
        model: HookedTransformer,
        cache: ActivationCache,
        config,
    ):
        # make n_components, batch, pos, d_model by expanding
        preds = torch.stack([outputs[key] for key in sorted(outputs.keys())], dim=0)

        # apply ln
        ln_applied_pred = cache.apply_ln_to_stack(
            preds, layer=None, pos_slice=config.pos_slice
        )
        unembed = model.unembed.W_U.T  # shape: d_vocab_out, d_model
        unembed_applied = einsum(
            "... d_model, d_vocab_out d_model -> ... d_vocab_out",
            ln_applied_pred,
            unembed,
        )

        # convert back to dictionary
        out_in_dict = {}
        for i, key in enumerate(sorted(outputs.keys())):
            out_in_dict[key] = unembed_applied[i]

        return out_in_dict


def make_linear_map(d_model: int, config: TunedLensConfig):
    model = nn.Linear(d_model, d_model).to(config.device)
    model.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    return {
        "model": model,
        "optimizer": optimizer,
    }
