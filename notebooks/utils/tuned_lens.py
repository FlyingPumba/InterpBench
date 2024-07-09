
from dataclasses import dataclass
from dataclasses import field
import torch.nn as nn
import torch
from fancy_einsum import einsum
from iit.model_pairs.base_model_pair import BaseModelPair
from transformer_lens.ActivationCache import ActivationCache
from tqdm import tqdm
from transformer_lens import HookedTransformer
import gc
from iit.utils import index
from iit.utils.metric import MetricStoreCollection, MetricStore, MetricType

@dataclass
class TunedLensConfig:
    """
    Configuration for trainin tuned lens maps

    Args:
        num_epochs: number of epochs to train
        lr: learning rate
        from_activation: whether to train from activations. If False, train from resid stacks (default: True)
        to_logits: whether to train to logits. If False, train to hook_resid_post of final layer (default: True)
        pos_slice: slice to apply to the positional dimension. Default is to exclude the BOS token (slice(1, None, None))
    """

    num_epochs: int
    lr: float
    from_activation: bool = False
    to_logits: bool = True
    pos_slice: slice = field(default_factory=lambda: slice(1, None, None))
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    loss: nn.Module = nn.MSELoss()
    do_per_vocab: bool = False

    def __post_init__(self):
        self.pos_idx = index.TorchIndex([
            slice(None, None, None),
            self.pos_slice,
        ]).as_index


def train_translators(
        translators: 'Translators',
        model: HookedTransformer,
        loader: torch.utils.data.DataLoader, 
        config: TunedLensConfig):
    # train each probe
    translators.train()
    
    per_epoch_metrics = []
    for epoch in tqdm(range(config.num_epochs)):
        metricstore = MetricStoreCollection([
            MetricStore(f"{key}", MetricType.LOSS) for key in translators.keys()
        ])
        for batch in loader:
            gc.collect()
            if config.device == "cuda":
                torch.cuda.empty_cache()
            with torch.no_grad():
                original_logits, cache = model.run_with_cache(batch)
                original_logits = original_logits[config.pos_idx]
            translators.zero_grad()

            translator_outs = translators.forward(cache, config.pos_slice)
            if config.to_logits:
                preds = Translators.apply_last_layer_to_translator_outputs(translator_outs, model, cache, config)
                labels = original_logits
            else:
                preds = translator_outs
                labels = cache[f"blocks.{model.cfg.n_layers-1}.hook_resid_post"]
            losses = {}
            loss = 0
            for key, pred in preds.items():
                # print(key, pred.requires_grad, translators[key]["model"].weight.grad is not None)
                _loss = config.loss(pred, labels.clone().detach())
                loss += _loss
                losses[key] = _loss.item()
            loss.backward()
            translators.step()
            metricstore.update(losses)
        per_epoch_metrics.append(metricstore.to_dict())
    return per_epoch_metrics
    

def do_tuned_lens(
    model_pair: BaseModelPair,
    loader: torch.utils.data.DataLoader,
    config: TunedLensConfig = TunedLensConfig(1, 1e-3),
    return_train_metrics: bool = False
):
    # make translator linear maps for each node
    model = model_pair.ll_model
    num_heads = model.cfg.n_heads
    num_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    translators = Translators.make_translators(d_model, num_heads, num_layers, config)

    # train translators
    train_metrics =train_translators(translators, model, loader, config)
    
    # do tuned lens
    translators.eval()
    tuned_lens_results = {}
    labels = None
    with torch.no_grad():
        for batch in loader:
            original_logits, cache = model.run_with_cache(batch)
            original_logits = original_logits[config.pos_idx]
            translator_outs = translators.forward(cache, config.pos_slice)
            preds = Translators.apply_last_layer_to_translator_outputs(translator_outs, model, cache, config)
            for key, pred in preds.items():
                if key not in tuned_lens_results:
                    tuned_lens_results[key] = pred
                else:
                    tuned_lens_results[key] = torch.cat([tuned_lens_results[key], pred], dim=0)
            if labels is not None:
                labels = torch.cat([labels, original_logits], dim=0)
            else:
                labels = original_logits
    if return_train_metrics:
        return tuned_lens_results, labels, train_metrics
    return tuned_lens_results, labels
            
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
        preds ={}
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
        config):
        # make n_components, batch, pos, d_model by expanding
        preds = torch.stack([outputs[key] for key in sorted(outputs.keys())], dim=0)

        # apply ln
        ln_applied_pred = cache.apply_ln_to_stack(preds, layer=None, pos_slice=config.pos_slice)
        unembed = model.unembed.W_U.T # shape: d_vocab_out, d_model
        unembed_applied = einsum(
            "... d_model, d_vocab_out d_model -> ... d_vocab_out",
            ln_applied_pred,
            unembed
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
                