import torch

import iit.model_pairs as mp
from iit.utils.node_picker import (
    find_ll_node_by_name,
    get_all_individual_nodes_in_circuit,
    get_nodes_not_in_circuit,
)
from interp_utils.node_stats.stats_to_df import stats_to_df


def get_node_norm_stats(model_pair, loader, return_cache_dict=False):
    cache_dict = {}
    nodes_not_in_circuit = get_nodes_not_in_circuit(
        model_pair.ll_model, model_pair.corr
    )
    nodes_in_circuit = get_all_individual_nodes_in_circuit(
        model_pair.ll_model, model_pair.corr
    )
    model = model_pair.ll_model

    for batch in loader:
        _, cache = model.run_with_cache(batch[0])
        for node, tensor in cache.items():
            nodes_found_in_circuit = find_ll_node_by_name(node, nodes_in_circuit)
            nodes_found_not_in_circuit = find_ll_node_by_name(
                node, nodes_not_in_circuit
            )
            if len(nodes_found_in_circuit) > 0:
                for node in nodes_found_in_circuit:
                    mean, norm = get_mean_and_norm(cache, node)
                    if node not in cache_dict:
                        cache_dict[node] = {
                            "mean_cache": mean / len(loader),
                            "norm_cache": norm,
                            "in_circuit": True,
                        }
                    else:
                        cache_dict[node]["mean_cache"] += mean / len(loader)
                        cache_dict[node]["norm_cache"].extend(norm)
            if len(nodes_found_not_in_circuit) > 0:
                for node in nodes_found_not_in_circuit:
                    mean, norm = get_mean_and_norm(cache, node)
                    if node not in cache_dict:
                        cache_dict[node] = {
                            "mean_cache": mean / len(loader),
                            "norm_cache": norm,
                            "in_circuit": False,
                        }
                    else:
                        cache_dict[node]["mean_cache"] += mean / len(loader)
                        cache_dict[node]["norm_cache"].extend(norm)

    for node, cache in cache_dict.items():
        cache_dict[node]["norm_std"] = torch.std(
            torch.tensor(cache["norm_cache"])
        ).item()
        cache_dict[node]["norm_cache"] = sum(cache["norm_cache"]) / len(
            cache["norm_cache"]
        )
    if return_cache_dict:
        return node_norm_stats_to_df(cache_dict), cache_dict
    return node_norm_stats_to_df(cache_dict)


def get_mean_and_norm(cache, node: mp.LLNode):
    cache_val = cache[node.name]
    cache_slice = cache_val[node.index.as_index]
    return cache_slice.mean(dim=0).unsqueeze(0), [
        torch.norm(t, p=2).item() for t in list(cache_slice)
    ]

def node_norm_stats_to_df(cache_dict):
    node_norms = stats_to_df(cache_dict, ["in_circuit", "norm_cache", "norm_std"])
    node_norms = node_norms.sort_values(by="in_circuit", ascending=True)
    return node_norms
