from networkx.classes.reportviews import NodeView

from circuits_benchmark.transformers.circuit_node import CircuitNode


class CircuitNodeView(NodeView):
  def __contains__(self, item: str | CircuitNode):
    if isinstance(item, str):
      return any([item == node.name for node in self._nodes])
    elif isinstance(item, CircuitNode):
      return any([item == node for node in self._nodes])
    else:
      return False
