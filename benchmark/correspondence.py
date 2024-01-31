# %%

import pickle
from networkx import Graph, DiGraph
import networkx as nx
import matplotlib.pyplot as plt

import traceback

import numpy as np
import torch as t
from typing import Callable

from commands.compile_benchmark import build_tracr_model
from benchmark.benchmark_case import BenchmarkCase
from tracr.compiler import compiling
from tracr.compiler.assemble import AssembledTransformerModel
from tracr.compiler.compiling import TracrOutput
from tracr.transformer.encoder import CategoricalEncoder
from tracr.craft.transformers import MLP, MultiAttentionHead, SeriesWithResiduals
from tracr.craft import bases
from utils.get_cases import get_cases
from utils.hooked_tracr_transformer import HookedTracrTransformer
from submodules.iit.model_pairs import HLNode

# %%

def names(vsb: bases.VectorSpaceWithBasis):
  return [(bd.name, bd.value) for bd in vsb.basis]

def print_craft_model(craft_model: SeriesWithResiduals):
  for block in craft_model.blocks:
      if isinstance(block, MLP):
          print("MLP:")
          print(names(block.fst.input_space), " -> ", names(block.fst.output_space))
          assert block.fst.output_space == block.snd.input_space
          print("\t -> ", names(block.snd.output_space))

      elif isinstance(block, MultiAttentionHead):
          print("MultiAttentionHead:")
          for i, sb in enumerate(block.sub_blocks):
            print(f"head {i}:")
            print(f"qk left {names(sb.w_qk.left_space)}")
            print(f"qk right {names(sb.w_qk.right_space)}")
            w_ov = sb.w_ov
            print(f"w_ov {names(w_ov.input_space)} -> \n\t{names(w_ov.output_space)}")

      else:
        raise ValueError(f"Unknown block type {type(block)}")
      print()

# %%

# %%
# with open("benchmark/case-00003/tracr_graph.pkl", "rb") as f:
#     graph3 = pickle.load(f)

# with open("benchmark/case-00003/tracr_model.pkl", "rb") as f:
#     model3 = pickle.load(f)

class TracrCorrespondence:
  """
  Stores a dictionary that takes tracr graph nodes to HookPoint nodes...
  """
  def __init__(self, graph: nx.DiGraph, craft_model: SeriesWithResiduals):
    self.graph = graph
    self.craft_model = craft_model

    self._dict = dict()

    for i, block in enumerate(craft_model.blocks):
      if isinstance(block, MLP):
          assert block.fst.output_space == block.snd.input_space
          for direction in block.snd.output_space.basis:
            self._dict[direction] = (i, "mlp", None)
      elif isinstance(block, MultiAttentionHead):
          for j, sb in enumerate(block.sub_blocks):
            for direction in sb.w_ov.output_space.basis:
              self._dict[direction] = (i, "attn", j)
      else:
        raise ValueError(f"Unknown block type {type(block)}")
    
    """
    This is not necessary for now, because node names and direction names are equal
    """
    # for name, node in graph.nodes.items():
    #   if name in ["indices", "tokens"]: continue
    #   if "OUTPUT_BASIS" in node:
    #     for direction in node["OUTPUT_BASIS"]:
    #       self._dict[direction] = self._unit_output_bases[direction]

  def __getitem__(self, key):
    return self._dict[key]
  
  def __repr__(self) -> str:
    return f"TracrCorrespondence(\n{self._dict}\n)"


tracr_output = build_tracr_model(BenchmarkCase("benchmark/case-00003"), force=True)
graph3 = tracr_output.graph
craft_model3 = tracr_output.craft_model
corr = TracrCorrespondence(graph3, craft_model3)
print(corr)

# %%
