from dataclasses import dataclass
from dataclasses import field
import torch.nn as nn
import torch
from iit.utils import index

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