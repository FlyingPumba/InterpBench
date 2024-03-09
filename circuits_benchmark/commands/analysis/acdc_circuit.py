from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.transformers.acdc_circuit_builder import get_full_acdc_circuit
from circuits_benchmark.transformers.circuit import Circuit
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

  calculate_fpr_and_tpr(acdc_circuit, case, verbose=args.verbose)


def calculate_fpr_and_tpr(acdc_circuit: Circuit,
                          case: BenchmarkCase,
                          verbose: bool = False):
  full_circuit = get_full_acdc_circuit(case.get_tl_model().cfg.n_layers)
  all_nodes = set(full_circuit.nodes)
  all_edges = set(full_circuit.edges)

  tracr_hl_circuit, tracr_ll_circuit, alignment = case.get_tracr_circuit(granularity="acdc_hooks")

  # remove from ACDC nodes and edges the indices at the end (e.g., "[:]") to compare with tracr
  acdc_nodes = [node.split("[")[0] for node in acdc_circuit.nodes]
  acdc_edges = [(edge[0].split("[")[0], edge[1].split("[")[0]) for edge in acdc_circuit.edges]

  # calculate nodes false positives and false negatives
  acdc_nodes = set(acdc_nodes)
  tracr_nodes = set(tracr_ll_circuit.nodes)
  false_positive_nodes = acdc_nodes - tracr_nodes
  false_negative_nodes = tracr_nodes - acdc_nodes
  true_positive_nodes = acdc_nodes & tracr_nodes
  true_negative_nodes = all_nodes - (acdc_nodes | tracr_nodes)

  if verbose:
    print("\nNodes analysis:")
    print(f" - False Positives: {sorted(false_positive_nodes)}")
    print(f" - False Negatives: {sorted(false_negative_nodes)}")
    print(f" - True Positives: {sorted(true_positive_nodes)}")
    print(f" - True Negatives: {sorted(true_negative_nodes)}")

  # calculate edges false positives and false negatives
  acdc_edges = set(acdc_edges)
  tracr_edges = set(tracr_ll_circuit.edges)
  false_positive_edges = acdc_edges - tracr_edges
  false_negative_edges = tracr_edges - acdc_edges
  true_positive_edges = acdc_edges & tracr_edges
  true_negative_edges = all_edges - (acdc_edges | tracr_edges)

  if verbose:
    print("\nEdges analysis:")
    print(f" - False Positives: {sorted(false_positive_edges)}")
    print(f" - False Negatives: {sorted(false_negative_edges)}")
    print(f" - True Positives: {sorted(true_positive_edges)}")
    print(f" - True Negatives: {sorted(true_negative_edges)}")

  # print FP and TP rates for nodes and edges as summary
  print(f"\nSummary:")

  nodes_tpr = len(true_positive_nodes) / len(true_positive_nodes | false_negative_nodes)
  print(f" - Nodes TP rate: {nodes_tpr}")
  nodes_fpr = len(false_positive_nodes) / len(false_positive_nodes | true_negative_nodes)
  print(f" - Nodes FP rate: {nodes_fpr}")

  edges_tpr = len(true_positive_edges) / len(true_positive_edges | false_negative_edges)
  print(f" - Edges TP rate: {edges_tpr}")
  edges_fpr = len(false_positive_edges) / len(false_positive_edges | true_negative_edges)
  print(f" - Edges FP rate: {edges_fpr}")

  return {
    "nodes": {
      "true_positive": true_positive_nodes,
      "false_positive": false_positive_nodes,
      "false_negative": false_negative_nodes,
      "true_negative": true_negative_nodes,
      "tpr": nodes_tpr,
      "fpr": nodes_fpr,
    },
    "edges": {
      "true_positive": true_positive_edges,
      "false_positive": false_positive_edges,
      "false_negative": false_negative_edges,
      "true_negative": true_negative_edges,
      "tpr": edges_tpr,
      "fpr": edges_fpr,
    },
  }