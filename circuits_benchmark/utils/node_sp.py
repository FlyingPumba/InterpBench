from acdc.docstring.utils import AllDataThings
from subnetwork_probing.train import NodeLevelMaskedTransformer
from subnetwork_probing.train import train_sp as train_node_sp


def train_sp(
    args,
    masked_model: NodeLevelMaskedTransformer,
    all_task_things: AllDataThings,
):
    return train_node_sp(
        args=args,
        masked_model=masked_model,
        all_task_things=all_task_things,
    )
