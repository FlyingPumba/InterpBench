import gc

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from iit.model_pairs.base_model_pair import BaseModelPair
from iit.utils.metric import MetricStore, MetricStoreCollection, MetricType
from interp_utils.lens import Translators, TunedLensConfig


def train_translators(
    translators: "Translators",
    model: HookedTransformer,
    loader: torch.utils.data.DataLoader,
    config: TunedLensConfig,
):
    # train each probe
    translators.train()

    per_epoch_metrics = []
    for epoch in tqdm(range(config.num_epochs)):
        metricstore = MetricStoreCollection(
            [MetricStore(f"{key}", MetricType.LOSS) for key in translators.keys()]
        )
        for batch in loader:
            gc.collect()
            if config.device == "cuda":
                torch.cuda.empty_cache()
            with torch.no_grad():
                original_logits, cache = model.run_with_cache(batch[0])
                original_logits = original_logits[config.pos_idx]
            translators.zero_grad()

            translator_outs = translators.forward(cache, config.pos_slice)
            if config.to_logits:
                preds = Translators.apply_last_layer_to_translator_outputs(
                    translator_outs, model, cache, config
                )
                labels = original_logits
            else:
                preds = translator_outs
                labels = cache[f"blocks.{model.cfg.n_layers-1}.hook_resid_post"]
            losses = {}
            loss = 0
            for key, pred in preds.items():
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
    return_train_metrics: bool = False,
):
    # make translator linear maps for each node
    model = model_pair.ll_model
    num_heads = model.cfg.n_heads
    num_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    translators = Translators.make_translators(d_model, num_heads, num_layers, config)

    # train translators
    train_metrics = train_translators(translators, model, loader, config)

    # do tuned lens
    translators.eval()
    tuned_lens_results = {}
    labels = None
    with torch.no_grad():
        for batch in loader:
            original_logits, cache = model.run_with_cache(batch[0])
            original_logits = original_logits[config.pos_idx]
            translator_outs = translators.forward(cache, config.pos_slice)
            preds = Translators.apply_last_layer_to_translator_outputs(
                translator_outs, model, cache, config
            )
            for key, pred in preds.items():
                if key not in tuned_lens_results:
                    tuned_lens_results[key] = pred
                else:
                    tuned_lens_results[key] = torch.cat(
                        [tuned_lens_results[key], pred], dim=0
                    )
            if labels is not None:
                labels = torch.cat([labels, original_logits], dim=0)
            else:
                labels = original_logits
    if return_train_metrics:
        return tuned_lens_results, labels, train_metrics
    return tuned_lens_results, labels
