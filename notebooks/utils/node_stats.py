import iit.model_pairs as mp
import torch
from iit.utils.node_picker import get_nodes_not_in_circuit, get_nodes_in_circuit, get_all_individual_nodes_in_circuit
import pandas as pd


def find_ll_node_by_name(name, list_of_nodes):
    ll_nodes = []
    for node in list_of_nodes:
        if node.name == name:
            ll_nodes.append(node)
    return ll_nodes


def get_mean_and_norm(cache, node: mp.LLNode):
    cache_val = cache[node.name]
    cache_slice = cache_val[node.index.as_index]
    return cache_slice.mean(dim=0).unsqueeze(0), [
        torch.norm(t, p=2).item() for t in list(cache_slice)
    ]


def get_node_stats(model_pair, loader):
    cache_dict = {}
    nodes_not_in_circuit = get_nodes_not_in_circuit(
        model_pair.ll_model, model_pair.corr
    )
    nodes_in_circuit = get_all_individual_nodes_in_circuit(model_pair.ll_model, model_pair.corr)
    model = model_pair.ll_model

    for batch in loader:
        _, cache = model.run_with_cache(batch)
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
    return cache_dict


def node_stats_to_df(cache_dict):
    node_norms = pd.DataFrame(columns=["name", "in_circuit", "norm_cache", "norm_std"])
    for k, v in cache_dict.items():
        # print(f"{v['in_circuit']} {v['norm_cache']} {v['norm_std']}")
        entry = {
            "name": (
                (k.name + f", head {str(k.index).split(',')[2]}")
                if "attn" in k.name
                else k.name
            ),
            "in_circuit": v["in_circuit"],
            "norm_cache": v["norm_cache"],
            "norm_std": v["norm_std"],
        }
        node_norms = pd.concat(
            [node_norms, pd.DataFrame(entry, index=[0])], axis=0, ignore_index=True
        )

    node_norms = node_norms.sort_values(by="in_circuit", ascending=True)
    return node_norms