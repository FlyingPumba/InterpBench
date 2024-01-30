# %%

import pickle
from networkx import Graph
import networkx as nx
import matplotlib.pyplot as plt

import traceback

import numpy as np
import torch
import torch as t

from benchmark.benchmark_case import BenchmarkCase
from compression.residual_stream import compress, setup_compression_training_args_for_parser
from tracr.compiler import compiling
from tracr.compiler.assemble import AssembledTransformerModel
from tracr.compiler.compiling import TracrOutput
from tracr.transformer.encoder import CategoricalEncoder
from utils.get_cases import get_cases
from utils.hooked_tracr_transformer import HookedTracrTransformer

# %%

def build_tracr_model(case: BenchmarkCase, force: bool = False) -> TracrOutput:
  """Compiles a single case to a tracr model."""

  # if tracr model and output have already compiled, we just load and return them
  if not force:
    tracr_model = case.load_tracr_model()
    tracr_graph = case.load_tracr_graph()
    if tracr_model is not None and tracr_graph is not None:
      return TracrOutput(tracr_model, tracr_graph)

  program = case.get_program()
  max_seq_len = case.get_max_seq_len()
  vocab = case.get_vocab()

  tracr_output = compiling.compile_rasp_to_model(
    program,
    vocab=vocab,
    max_seq_len=max_seq_len,
    compiler_bos="BOS",
    return_craft_model=True
  )

  return tracr_output

# %%

cases = get_cases(None)
case3 = cases[3-1]
tracr_output = build_tracr_model(case3, force=True)
# %%
params = list(tracr_output.model.params.keys())
params
# %%
for block in tracr_output.craft_model.blocks:
    # for attr in dir(block):
    #     if not attr.startswith("_"):
    #         print(f"{attr}: {getattr(block, attr)}")

    if 'sub_blocks' in dir(block):
      for sb in block.sub_blocks:
        print(sb)
        for attr in dir(sb):
            if not attr.startswith("_"):
                print(f"\t{attr}: {getattr(sb, attr)}")

    print()

subblock_dict = dict()
for block in tracr_output.craft_model.blocks:
    if 'sub_blocks' in dir(block):
      for i, sb in enumerate(block.sub_blocks):
        hashable = sb.__dict__.values()
        subblock_dict[hashable] = (block, i)
# %%

# %%
# with open("benchmark/case-00003/tracr_graph.pkl", "rb") as f:
#     graph3 = pickle.load(f)

# with open("benchmark/case-00003/tracr_model.pkl", "rb") as f:
#     model3 = pickle.load(f)
    
graph3 = tracr_output.graph
craft_model3 = tracr_output.craft_model

# %%

len(craft_model3.blocks)
# %%
# Draw the graph
nx.draw(graph3, with_labels=True)

# Show the graph
plt.show()
# %%

for node in graph3.nodes:
    print(node)
    node_obj = graph3.nodes[node]
    if "OUTPUT_BASIS" in node_obj:
       print(f"node {node} has OUTPUT_BASIS")
       ob = node_obj["OUTPUT_BASIS"]
    
    for k, v in graph3.nodes[node].items():
        if k != "MODEL_BLOCK":
          print(f"\t{k}: {v}")

    if "MODEL_BLOCK" in node_obj:
      block = node_obj["MODEL_BLOCK"]
      if block is not None and block.__dict__.values() in subblock_dict:
        print(f"block is in craft_model3 in {subblock_dict[block]}")
    print()
# %%

# %%
craft_model3

# %%

# %%
