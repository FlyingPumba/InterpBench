from circuits_benchmark.transformers.acdc_circuit_builder import get_full_acdc_circuit
from circuits_benchmark.transformers.circuit import Circuit
from circuits_benchmark.utils.circuits_comparison import calculate_fpr_and_tpr
from circuits_benchmark.utils.get_cases import get_cases


def setup_args_parser(subparsers):
  parser = subparsers.add_parser("acdc-circuit")
  parser.add_argument("-i", type=str, required=True,
                      help="The index of the case to compare against.")
  parser.add_argument("--path", type=str, required=True,
                      help="The path to the acdc circuit to analyze (.pkl file).")
  parser.add_argument("-v", "--verbose", action="store_true",
                      help="Prints more information about the process.")


def run(args):
  acdc_circuit = Circuit.load(args.path)
  if acdc_circuit is None:
    raise ValueError(f"Failed to load acdc circuit from {args.path}")

  cases = get_cases(indices=[args.i])
  if len(cases) == 0:
    raise ValueError(f"No case found with index {args.i}")
  case = cases[0]

  full_circuit = get_full_acdc_circuit(case.get_tl_model().cfg.n_layers, case.get_tl_model().cfg.n_heads)
  tracr_hl_circuit, tracr_ll_circuit, alignment = case.get_tracr_circuit(granularity="acdc_hooks")

  calculate_fpr_and_tpr(acdc_circuit, tracr_ll_circuit, full_circuit, verbose=args.verbose)
