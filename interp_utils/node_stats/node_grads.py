import numpy as np
import torch
from tqdm import tqdm

import iit.model_pairs as mp
from iit.utils.node_picker import (
    find_ll_node_by_name,
    get_all_individual_nodes_in_circuit,
    get_nodes_not_in_circuit,
)

from .stats_to_df import stats_to_df


def get_grad_norms(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: callable,
):
    grad_norms = {}
    param_grad_norms = {}
    losses = []
    for x, y in tqdm(loader):
        logits, cache = model.run_with_cache(x)
        loss = loss_fn(logits, y)
        model.zero_grad()
        loss.backward()
        losses.append(loss.item())
        for k, v in cache.items():
            if k not in grad_norms:
                grad_norms[k] = v.grad.mean(dim=0) / len(loader)
            else:
                grad_norms[k] += v.grad.mean(dim=0) / len(loader)

        for k, v in model.named_parameters():
            if k not in param_grad_norms:
                param_grad_norms[k] = v.grad.mean(dim=0) / len(loader)
            else:
                param_grad_norms[k] += v.grad.mean(dim=0) / len(loader)

    for k, v in grad_norms.items():
        grad_norms[k] = v.norm().item()
    for k, v in param_grad_norms.items():
        param_grad_norms[k] = v.norm().item()

    return {
        "grad_norms": grad_norms,
        "param_grad_norms": param_grad_norms,
        "loss": np.mean(losses),
    }


def get_grad_norms_by_node(
    model_pair: mp.BaseModelPair,
    loader: torch.utils.data.DataLoader,
    loss_fn: callable,
    return_cache_dict=False,
):
    cache_dict = {}
    nodes_not_in_circuit = get_nodes_not_in_circuit(
        model_pair.ll_model, model_pair.corr
    )
    nodes_in_circuit = get_all_individual_nodes_in_circuit(
        model_pair.ll_model, model_pair.corr
    )
    model = model_pair.ll_model
    # turn on grads
    model.train()
    model.requires_grad_(True)
    model = mp.ll_model.LLModel(model, detach_while_caching=False)
    
    for batch in tqdm(loader):
        logits, cache = model.run_with_cache(batch[0])
        loss = loss_fn(logits, batch[1])
        model.zero_grad()
        loss.backward()
        for node, tensor in cache.items():
            nodes_found_in_circuit = find_ll_node_by_name(node, nodes_in_circuit)
            nodes_found_not_in_circuit = find_ll_node_by_name(
                node, nodes_not_in_circuit
            )
            if len(nodes_found_in_circuit) > 0:
                for node in nodes_found_in_circuit:
                    if node not in cache_dict:
                        cache_dict[node] = {
                            "grad_cache": tensor.grad.mean(dim=0) / len(loader),
                            "in_circuit": True,
                        }
                    else:
                        cache_dict[node]["grad_cache"] += tensor.grad.mean(dim=0) / len(loader)
            if len(nodes_found_not_in_circuit) > 0:
                for node in nodes_found_not_in_circuit:
                    if node not in cache_dict:
                        cache_dict[node] = {
                            "grad_cache": tensor.grad.mean(dim=0) / len(loader),
                            "in_circuit": False,
                        }
                    else:
                        cache_dict[node]["grad_cache"] += tensor.grad.mean(dim=0) / len(loader)
    for node, cache in cache_dict.items():
        cache_dict[node]["grad_norm"] = cache["grad_cache"].norm().item()
        cache_dict[node]["grad_std"] = cache["grad_cache"].std().item()

    grad_stats = stats_to_df(cache_dict, ["in_circuit", "grad_norm", "grad_std"])
    grad_stats = grad_stats.sort_values(by="in_circuit", ascending=True)

    if return_cache_dict:
        return grad_stats, cache_dict
    return grad_stats
