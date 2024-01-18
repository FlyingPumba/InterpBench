import traceback

from benchmark.benchmark_case import BenchmarkCase
from tracr.compiler import compiling
from tracr.compiler.compiling import TracrOutput
from utils.get_cases import get_cases
from utils.hooked_tracr_transformer import HookedTracrTransformer


def setup_args_parser(subparsers):
  compile_parser = subparsers.add_parser("compile")
  compile_parser.add_argument("-i", "--indices", type=str, default=None,
                              help="A list of comma separated indices of the cases to compile. "
                                   "If not specified, all cases will be compiled.")
  compile_parser.add_argument("-f", "--force", action="store_true",
                              help="Force compilation of cases, even if they have already been compiled.")


def compile_all(args):
  for case in get_cases(args):
    print(f"\nCompiling {case}")
    try:
      tracr_output = build_tracr_model(case, args.force)
      build_transformer_lens_model(case, args.force, tracr_output=tracr_output)
    except Exception as e:
      print(f" >>> Failed to compile {case}:")
      traceback.print_exc()
      continue


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
  )

  # write tracr model and graph to disk
  case.dump_tracr_model(tracr_output.model)
  case.dump_tracr_graph(tracr_output.graph)

  return tracr_output


def build_transformer_lens_model(case: BenchmarkCase,
                                 force: bool = False,
                                 tracr_output: TracrOutput = None) -> HookedTracrTransformer:
  """Compiles a tracr model to transformer lens."""
  if not force:
    tl_model = case.load_tl_model()
    if tl_model is not None:
      return tl_model

  tracr_model = None
  if tracr_output is not None:
    tracr_model = tracr_output.model

  if tracr_model is None:
    tracr_model = case.load_tracr_model()

  if tracr_model is None:
    tracr_output = build_tracr_model(case, force)
    tracr_model = tracr_output.model

  tl_model = HookedTracrTransformer(tracr_model)

  case.dump_tl_model(tl_model)

  return tl_model
