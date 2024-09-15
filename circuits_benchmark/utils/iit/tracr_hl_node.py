from iit.utils import index
from iit.utils.nodes import HLNode

from circuits_benchmark.utils.circuit.circuit_node import CircuitNode


class TracrHLNode(HLNode):
    def __init__(self, name, label, num_classes, idx=None):
        # type checks
        assert isinstance(name, str), ValueError(
            f"name is not a string, but {type(name)}"
        )
        assert isinstance(label, str), ValueError(
            f"label is not a string, but {type(label)}"
        )
        assert isinstance(num_classes, int), ValueError(
            f"num_classes is not an int, but {type(num_classes)}"
        )
        assert idx is None or isinstance(idx, index.TorchIndex), ValueError(
            f"index is not a TorchIndex, but {type(index)}"
        )
        super().__init__(name, num_classes, idx)
        self.label = label

    def get_label(self) -> str:
        return self.label

    def get_name(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return super().__hash__() + hash(self.label)

    def __str__(self) -> str:
        return super().__str__()

    def __eq__(self, other) -> bool:
        if isinstance(other, TracrHLNode):
            return (
                self.name == other.name
                and self.label == other.label
                and self.num_classes == other.num_classes
                and self.index == other.index
            )
        if isinstance(other, CircuitNode):
            if "attn" in other.name:
                other_idx = index.Ix[:, :, other.index, :]
                return self.name == other.name and self.index == other_idx
            else:
                return self.name == other.name and (
                    self.index == index.Ix[[None]] or self.index == None
                )
        return super().__eq__(other)

    def __repr__(self) -> str:
        return f"TracrHLNode(name: {self.name},\n label: {self.label},\n classes: {self.num_classes},\n index: {self.index}\n)"

    @classmethod
    def from_tracr_node(cls, tracr_node=CircuitNode, label="", num_classes=-1):
        name = tracr_node.name
        tracr_index = tracr_node.index

        if "mlp" in name:
            return TracrHLNode(
                name=tracr_node.name,
                label=label,
                num_classes=num_classes,
            )
        else:
            return TracrHLNode(
                name=tracr_node.name,
                label=label,
                num_classes=num_classes,
                idx=index.Ix[:, :, tracr_index, :],
            )
