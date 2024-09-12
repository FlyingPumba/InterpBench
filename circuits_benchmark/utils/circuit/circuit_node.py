class CircuitNode(object):
    def __init__(self, name: str, index: int | None = None):
        self.name = name
        self.index = index

    def __str__(self):
        return f"{self.name}[{self.index}]" if self.index is not None else self.name

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if not isinstance(other, CircuitNode):
            return False

        return self.name == other.name and self.index == other.index

    def __lt__(self, other):
        if not isinstance(other, CircuitNode):
            raise ValueError(f"Expected a CircuitNode, got {type(other)}")

        if self.name != other.name:
            return self.name < other.name
        elif self.index is None:
            return False
        elif other.index is None:
            return True
        else:
            return self.index < other.index
