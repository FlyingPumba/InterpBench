import os

from cloudpickle import cloudpickle
from networkx import DiGraph

from tracr.compiler.assemble import AssembledTransformerModel
from utils.hooked_tracr_transformer import HookedTracrTransformer

PROJECT_FOLDER = "circuits-benchmark"


class BenchmarkCase(object):
  def __init__(self, file_path_from_root: str):
    self.file_path_from_root = file_path_from_root

  def get_file_path_from_root(self) -> str:
    return self.file_path_from_root

  def get_module_name(self) -> str:
    """Returns the module name from the file path.
    Basically, we replace "/" with "." and remove the ".py" extension.
    """
    return self.file_path_from_root.replace("/", ".")[:-3]

  def __str__(self):
    return self.file_path_from_root

  def get_tracr_model_path_from_root(self) -> str:
    return self.file_path_from_root.replace("rasp.py", "tracr_model.pkl")

  def get_tracr_graph_path_from_root(self) -> str:
    return self.file_path_from_root.replace("rasp.py", "tracr_graph.pkl")

  def get_tl_model_path_from_root(self) -> str:
    return self.file_path_from_root.replace("rasp.py", "tl_model.pkl")

  def load_tracr_model(self) -> AssembledTransformerModel | None:
    """Loads the tracr model from disk, if it exists."""
    tracr_model_output_path = self.relativize_path(self.get_tracr_model_path_from_root())
    return self.load_from_pickle(tracr_model_output_path)

  def load_tracr_graph(self) -> DiGraph | None:
    """Loads the tracr graph from disk, if it exists."""
    tracr_graph_output_path = self.relativize_path(self.get_tracr_graph_path_from_root())
    return self.load_from_pickle(tracr_graph_output_path)

  def load_tl_model(self) -> HookedTracrTransformer | None:
    """Loads the transformer_lens model from disk, if it exists."""
    tl_model_output_path = self.relativize_path(self.get_tl_model_path_from_root())
    return self.load_from_pickle(tl_model_output_path)

  def dump_tracr_model(self, tracr_model: AssembledTransformerModel) -> None:
    """Dumps the tracr model to disk."""
    tracr_model_output_path = self.relativize_path(self.get_tracr_model_path_from_root())
    self.dump_to_pickle(tracr_model_output_path, tracr_model)

  def dump_tracr_graph(self, tracr_graph: DiGraph) -> None:
    """Dumps the tracr graph to disk."""
    tracr_graph_output_path = self.relativize_path(self.get_tracr_graph_path_from_root())
    self.dump_to_pickle(tracr_graph_output_path, tracr_graph)

  def dump_tl_model(self, tl_model: HookedTracrTransformer) -> None:
    """Dumps the transformer_lens model to disk."""
    tl_model_output_path = self.relativize_path(self.get_tl_model_path_from_root())
    self.dump_to_pickle(tl_model_output_path, tl_model)

  def load_from_pickle(self, path) -> object | None:
    if os.path.exists(path):
      with open(path, "rb") as f:
        return cloudpickle.load(f)
    else:
      return None

  def dump_to_pickle(self, path, obj) -> None:
    with open(path, "wb") as f:
      cloudpickle.dump(obj, f)

  def relativize_path(self, path) -> str:
    """Relativizes the path to the project root."""
    cwd = os.getcwd()
    parts = cwd.split("/")
    parts.reverse()

    for part in parts:
      if part == PROJECT_FOLDER:
        break
      else:
        path = os.path.join("..", path)
    return path
