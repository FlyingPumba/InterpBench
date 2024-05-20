from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Optional

import cmapy
import networkx as nx
import numpy as np
import pygraphviz as pgv
from matplotlib import pyplot as plt
from networkx import DiGraph

from circuits_benchmark.transformers.circuit_granularity import CircuitGranularity
from circuits_benchmark.transformers.circuit_node import CircuitNode
from circuits_benchmark.transformers.circuit_node_view import CircuitNodeView
from circuits_benchmark.utils.cloudpickle import dump_to_pickle, load_from_pickle


class Circuit(DiGraph):
  def __init__(self, granularity: CircuitGranularity | None = None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.granularity = granularity

  def add_node(self, node_for_adding: CircuitNode, **attr):
    # Make sure that node_for_adding is a CircuitNode
    if not isinstance(node_for_adding, CircuitNode):
      raise ValueError(f"Expected a CircuitNode, got {type(node_for_adding)}")

    super().add_node(node_for_adding, **attr)

  def add_edge(self, u_of_edge: CircuitNode, v_of_edge: CircuitNode, **attr):
    # Make sure that u_of_edge and v_of_edge are CircuitNodes
    if not isinstance(u_of_edge, CircuitNode) or not isinstance(v_of_edge, CircuitNode):
      raise ValueError(f"Expected a CircuitNode, got {type(u_of_edge)} and {type(v_of_edge)}")

    super().add_edge(u_of_edge, v_of_edge, **attr)

  @cached_property
  def nodes(self):
    return CircuitNodeView(self)

  def save(self, file_path: str):
    if not file_path.endswith(".pkl"):
      file_path += ".pkl"

    dump_to_pickle(file_path, self)

  @staticmethod
  def load(file_path) -> Circuit | None:
    return load_from_pickle(file_path)

  def get_result_node(self):
    """Returns the node in the circuit that doesn't have successors (there should be only one)."""
    result_nodes = [node for node in self.nodes if not list(self.successors(node))]
    assert len(result_nodes) == 1, f"Expected 1 result node, got {len(result_nodes)}"
    return result_nodes[0]

  def nx_plot(self, file_path: str, seed: int=42, n_layers: Optional[int]=None, n_heads: Optional[int]=None):
    if n_layers is None or n_heads is None:
      pos = nx.spring_layout(self, seed=seed)
    else:
      pos = {}
      current_height = 0
      center_width = (n_heads * 3) // 2 - 1

      pos[CircuitNode("hook_embed")] = (center_width, current_height)
      pos[CircuitNode("hook_pos_embed")] = (center_width + 1, current_height)

      for layer in range(n_layers):
        attn_head_input_height = current_height - 1
        attn_head_height = current_height - 2
        attn_output_height = current_height - 3

        for head in range(n_heads):
          for letter, letter_index in zip(["q", "k", "v"], range(3)):
            node = CircuitNode(f"blocks.{layer}.hook_{letter}_input", head)
            if node in self.nodes:
              pos[node] = (letter_index + head * 3, attn_head_input_height)

          for letter, letter_index in zip(["q", "k", "v"], range(3)):
            node = CircuitNode(f"blocks.{layer}.attn.hook_{letter}", head)
            if node in self.nodes:
              pos[node] = (letter_index + head * 3, attn_head_height)

          node = CircuitNode(f"blocks.{layer}.attn.hook_result", head)
          if node in self.nodes:
            pos[node] = (center_width, attn_output_height)

        current_height = attn_output_height - 1
        node = CircuitNode(f"blocks.{layer}.hook_mlp_in")
        if node in self.nodes:
          pos[node] = (center_width, current_height)

        current_height -= 1
        node = CircuitNode(f"blocks.{layer}.hook_mlp_out")
        if node in self.nodes:
          pos[node] = (center_width, current_height)

      current_height -= 1
      node = CircuitNode(f"blocks.{n_layers-1}.hook_resid_post")
      if node in self.nodes:
        pos[node] = (center_width, current_height)

    # increase plt fig size
    plt.figure(figsize=(100,200))

    options = {
      "font_size": 16,
      "node_size": 20000,
      "node_color": "white",
      "edgecolors": "black",
      "width": 2,
    }
    nx.draw_networkx(self, pos, **options)

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

  def pgv_plot(
      self,
      file_path: str,
      minimum_penwidth: float = 1,
      seed: Optional[int] = 42
  ):
    node_colors = {}
    for node in self.nodes:
      node_colors[str(node)] = self.generate_random_color()

    g = pgv.AGraph(
      directed=True,
      bgcolor="transparent",
      overlap="false",
      splines="true",
      layout="dot",  # or neato
    )

    if seed is not None:
      np.random.seed(seed)

    # create all nodes and edges
    for parent, child in self.edges:
      parent_name = str(parent)
      child_name = str(child)

      for node_name in [parent_name, child_name]:
        g.add_node(
          node_name,
          fillcolor=node_colors[node_name],
          color="black",
          style="filled, rounded",
          shape="box",
          fontname="Helvetica",
        )

      edge_width = minimum_penwidth * 2

      g.add_edge(
        parent_name,
        child_name,
        penwidth=str(edge_width),
        color=node_colors[parent_name],
      )

    filename_without_extension = Path(file_path).stem
    g.write(path=filename_without_extension + ".gv")

    if not file_path.endswith(".gv"):  # turn the .gv file into a .png file
      g.draw(path=file_path, prog="dot")

  def generate_random_color(self) -> str:
    """
    https://stackoverflow.com/questions/28999287/generate-random-colors-rgb
    """
    return self.rgb2hex(cmapy.color("Pastel2", np.random.randint(0, 256), rgb_order=True))

  def rgb2hex(self, rgb):
    """
    https://stackoverflow.com/questions/3380726/converting-an-rgb-color-tuple-to-a-hexidecimal-string
    """
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])