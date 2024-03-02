from circuits_benchmark.transformers.circuit import Circuit
from circuits_benchmark.utils.get_cases import get_cases


def setup_args_parser(subparsers):
  parser = subparsers.add_parser("acdc-circuit")
  parser.add_argument("-i", type=str, required=True,
                      help="The index of the case to compare against.")
  parser.add_argument("--path", type=str, required=True,
                      help="The path to the acdc circuit to analyze (.pkl file).")


def run(args):
  acdc_circuit = Circuit.load(args.path)
  if acdc_circuit is None:
    raise ValueError(f"Failed to load acdc circuit from {args.path}")

  cases = get_cases(indices=[args.i])
  if len(cases) == 0:
    raise ValueError(f"No case found with index {args.i}")
  case = cases[0]

  tracr_hl_circuit, tracr_ll_circuit, alignment = case.get_tracr_circuit(granularity="acdc_hooks")

  # remove from ACDC nodes and edges the indices at the end (e.g., "[:]") to compare with tracr
  acdc_nodes = [node.split("[")[0] for node in acdc_circuit.nodes]
  acdc_edges = [(edge[0].split("[")[0], edge[1].split("[")[0]) for edge in acdc_circuit.edges]

  # calculate nodes false positives and false negatives
  print("\nNodes analysis:")
  acdc_nodes = set(acdc_nodes)
  tracr_nodes = set(tracr_ll_circuit.nodes)
  false_positives = acdc_nodes - tracr_nodes
  false_negatives = tracr_nodes - acdc_nodes
  print(f" - False Positives: {sorted(false_positives)}")
  print(f" - False Negatives: {sorted(false_negatives)}")

  # calculate edges false positives and false negatives
  print("\nEdges analysis:")
  acdc_edges = set(acdc_edges)
  tracr_edges = set(tracr_ll_circuit.edges)
  false_positives = acdc_edges - tracr_edges
  false_negatives = tracr_edges - acdc_edges
  print(f" - False Positives: {sorted(false_positives)}")
  print(f" - False Negatives: {sorted(false_negatives)}")

  # print rates of FP/FN for nodes and edges as summary
  print("\nSummary:")
  print(f" - Nodes FP rate: {len(false_positives) / len(tracr_nodes)}")
  print(f" - Nodes FN rate: {len(false_negatives) / len(tracr_nodes)}")
  print(f" - Edges FP rate: {len(false_positives) / len(tracr_edges)}")
  print(f" - Edges FN rate: {len(false_negatives) / len(tracr_edges)}")
