from subnetwork_probing.train import NodeLevelMaskedTransformer
from acdc.docstring.utils import AllDataThings
import torch
from subnetwork_probing.train import train_sp as train_node_sp

def train_sp(
    args,
    masked_model: NodeLevelMaskedTransformer,
    all_task_things: AllDataThings,
):
    # raise NotImplementedError("This function is not implemented yet. Use train_edge_sp instead.")
    return train_node_sp(
        args=args,
        masked_model=masked_model,
        all_task_things=all_task_things,
    )