import torch
from sklearn.decomposition import PCA

import iit.utils.index as index
from iit.utils.node_picker import find_ll_node_by_name, get_all_nodes


def collect_activations(model_pair, loader, pos_slice=slice(1, None, None)):
    activation_cache = {}
    nodes = get_all_nodes(model_pair.ll_model)
    pos_idx = index.TorchIndex(((slice(None), pos_slice))) if pos_slice is not None else index.Ix[slice(None)]
    for node in nodes:
        activation_cache[node] = None
    for batch in loader:
        _, batch_cache = model_pair.ll_model.run_with_cache(batch[0])
        for k, tensor in batch_cache.items():
            ll_node_for_k = find_ll_node_by_name(k, nodes)
            if len(ll_node_for_k) > 0:
                for node in ll_node_for_k:
                    act = tensor[node.index.as_index].cpu()[pos_idx.as_index]
                    if activation_cache[node] is None:
                        activation_cache[node] = act
                    else:
                        activation_cache[node] = torch.cat((activation_cache[node], act), dim=0).cpu()
    return activation_cache


def collect_pca_directions(activation_cache, num_pca_components=2):
    pca_dirs = {}

    for node, activations in activation_cache.items():
        # calculate pca directions for activations
        for i in range(activations.shape[1]):
            pca = PCA(n_components=num_pca_components)
            activations_to_pca = activations[:, i].detach().numpy()
            # center data before pca
            activations_to_pca = activations_to_pca - activations_to_pca.mean(axis=0)
            pca.fit(activations_to_pca)
            if pca_dirs.get(node) is None:
                pca_dirs[node] = {}
            pca_dirs[node][i] = pca.components_
    return pca_dirs