import cloudpickle
import importlib
import os
import traceback

from benchmark.defaults import default_max_seq_len, default_bos, default_vocab
from tracr.compiler import compiling
from tracr.compiler.compiling import TracrOutput
from utils.get_cases import get_cases_files
from utils.hooked_tracr_transformer import HookedTracrTransformer


def setup_args_parser(subparsers):
  compile_parser = subparsers.add_parser("compile")
  compile_parser.add_argument("-i", "--indices", type=str, default=None,
                              help="A list of comma separated indices of the cases to compile. "
                                   "If not specified, all cases will be compiled.")
  compile_parser.add_argument("-f", "--force", action="store_true",
                              help="Force compilation of cases, even if they have already been compiled.")


def compile(args):
  for file_path in get_cases_files(args):
    print(f"\nCompiling {file_path}")
    try:
      tracr_output = build_tracr_model(file_path, args.force)
    except Exception as e:
      print(f" >>> Failed to compile {file_path}:")
      traceback.print_exc()
      continue

def build_tracr_model(file_path: str, force_writing: bool = False) -> TracrOutput:
  """Compiles a single case to a tracr model."""
  tracr_model_output_path = file_path.replace("rasp.py", "tracr_model.pkl")
  tracr_graph_output_path = file_path.replace("rasp.py", "tracr_graph.pkl")

  # if both files exist, we don't need to compile, just load and return them
  if os.path.exists(tracr_model_output_path) and os.path.exists(tracr_graph_output_path):
    with open(tracr_model_output_path, "rb") as f:
      tracr_model = cloudpickle.load(f)
    with open(tracr_graph_output_path, "rb") as f:
      tracr_graph = cloudpickle.load(f)
    return TracrOutput(tracr_model, tracr_graph)

  # load modulle for the file
  module_name = file_path.replace("/", ".")[:-3]
  module = importlib.import_module(module_name)

  # evaluate "get_program()" method dinamically
  get_program_fn = getattr(module, 'get_program')
  program = get_program_fn()

  max_seq_len = default_max_seq_len
  if hasattr(module, 'get_max_seq_len'):
    get_max_seq_len_fn = getattr(module, 'get_max_seq_len')
    max_seq_len = get_max_seq_len_fn()

  vocab = default_vocab
  if hasattr(module, 'get_vocab'):
    get_vocab_fn = getattr(module, 'get_vocab')
    vocab = get_vocab_fn()

  tracr_output = compiling.compile_rasp_to_model(
    program,
    vocab=vocab,
    max_seq_len=max_seq_len,
    compiler_bos=default_bos,
  )

  # write to file if it doesn't exist or if we're forcing it
  if force_writing or not os.path.exists(tracr_model_output_path):
    with open(tracr_model_output_path, "wb") as f:
      cloudpickle.dump(tracr_output.model, f)

  if force_writing or not os.path.exists(tracr_graph_output_path):
    with open(tracr_graph_output_path, "wb") as f:
      cloudpickle.dump(tracr_output.graph, f)

  return tracr_output


def build_transformer_lens_model(file_path: str,
                                 force_writing: bool = False,
                                 tracr_output: TracrOutput = None) -> HookedTracrTransformer:
  """Compiles a tracr model to transformer lens."""
  tl_model_output_path = file_path.replace("rasp.py", "tl_model.pkl")
  if os.path.exists(tl_model_output_path) and not force_writing:
    with open(tl_model_output_path, "rb") as f:
      tl_model = cloudpickle.load(f)
    return tl_model

  # If we didn't pass in a tracr_output, try to load it from file
  if tracr_output is None:
    tracr_model_output_path = file_path.replace("rasp.py", "tracr_model.pkl")
    if os.path.exists(tracr_model_output_path):
      with open(tracr_model_output_path, "rb") as f:
        tracr_output = cloudpickle.load(f)
    else:
      tracr_output = build_tracr_model(file_path)

  tl_model = HookedTracrTransformer(tracr_output.model)

  if force_writing or not os.path.exists(tl_model_output_path):
    with open(tl_model_output_path, "wb") as f:
      cloudpickle.dump(tl_model, f)

  return tl_model
