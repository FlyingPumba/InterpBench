from typing import Optional

from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCEdge import EdgeType
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.circuit.circuit import Circuit
from circuits_benchmark.utils.circuit.circuit_node import CircuitNode
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
from circuits_benchmark.utils.iit._acdc_utils import get_gt_circuit
from circuits_benchmark.utils.iit.correspondence import TracrCorrespondence


def calculate_fpr_and_tpr(
    hypothesis_circuit: Circuit,  # e.g., the one provided by ACDC
    true_circuit: Circuit,  # e.g., the one provided by Tracr (ground truth)
    full_circuit: Circuit,
    verbose: bool = False,
    promote_to_heads: bool = True,
    print_summary: bool = True,
):
  if promote_to_heads:
    all_nodes = replace_inputs_and_qkv_nodes_with_outputs(full_circuit)
    true_nodes = replace_inputs_and_qkv_nodes_with_outputs(true_circuit)
    hypothesis_nodes = replace_inputs_and_qkv_nodes_with_outputs(hypothesis_circuit)
  else:
    all_nodes = full_circuit.nodes
    true_nodes = true_circuit.nodes
    hypothesis_nodes = hypothesis_circuit.nodes

  all_nodes = set(all_nodes)

  # calculate nodes false positives and false negatives
  hypothesis_nodes = set(hypothesis_nodes)
  true_nodes = set(true_nodes)

  assert hypothesis_nodes.issubset(
    all_nodes), f"hypothesis nodes contain the following nodes that are not in the full circuit: {hypothesis_nodes - all_nodes}"
  assert true_nodes.issubset(
    all_nodes), f"true nodes contain the following nodes that are not in the full circuit: {true_nodes - all_nodes}"

  false_positive_nodes = hypothesis_nodes - true_nodes
  false_negative_nodes = true_nodes - hypothesis_nodes
  true_positive_nodes = hypothesis_nodes & true_nodes
  true_negative_nodes = all_nodes - (hypothesis_nodes | true_nodes)

  if verbose:
    print("\nNodes analysis:")
    print(f" - False Positives: {sorted(false_positive_nodes)}")
    print(f" - False Negatives: {sorted(false_negative_nodes)}")
    print(f" - True Positives: {sorted(true_positive_nodes)}")
    print(f" - True Negatives: {sorted(true_negative_nodes)}")

  # calculate edges false positives and false negatives
  if promote_to_heads:
    hypothesis_edges = replace_inputs_and_qkv_edges_with_outputs(hypothesis_circuit)
    true_edges = replace_inputs_and_qkv_edges_with_outputs(true_circuit)
    all_edges = replace_inputs_and_qkv_edges_with_outputs(full_circuit)
  else:
    hypothesis_edges = hypothesis_circuit.edges
    true_edges = true_circuit.edges
    all_edges = full_circuit.edges

  hypothesis_edges = set(hypothesis_edges)
  true_edges = set(true_edges)
  all_edges = set(all_edges)

  assert hypothesis_edges.issubset(
    all_edges), f"hypothesis edges contain the following edges that are not in the full circuit: {hypothesis_edges - all_edges}, hypothesis edges: {hypothesis_edges}, all edges: {all_edges}"
  assert true_edges.issubset(
    all_edges), f"true edges contain the following edges that are not in the full circuit: {true_edges - all_edges}"

  false_positive_edges = (hypothesis_edges - true_edges) & all_edges
  false_negative_edges = true_edges - hypothesis_edges
  true_positive_edges = hypothesis_edges & true_edges
  true_negative_edges = all_edges - (
        hypothesis_edges | true_edges)  # == (all_edges - hypothesis_edges) & (all_edges - true_edges)

  if verbose:
    print("\nEdges analysis:")
    print(f" - False Positives: {sorted(false_positive_edges)}")
    print(f" - False Negatives: {sorted(false_negative_edges)}")
    print(f" - True Positives: {sorted(true_positive_edges)}")
    print(f" - True Negatives: {sorted(true_negative_edges)}")

  # print FP and TP rates for nodes and edges as summary
  make_summary = lambda *args, **kwargs: print(*args, **kwargs) if print_summary else None
  if verbose:
    make_summary("\n\n-------------------\n\nhypothesis_edges", hypothesis_edges, "\n-----------\n")
    make_summary("true_edges", true_edges, "\n-----------\n")
    make_summary("all_edges", all_edges, "\n\n-------------------\n\n")
  make_summary(f"\nSummary:")

  if len(true_positive_nodes | false_negative_nodes) == 0:
    nodes_tpr = "N/A"
    make_summary(f" - Nodes TP rate: N/A")
  else:
    nodes_tpr = len(true_positive_nodes) / len(true_positive_nodes | false_negative_nodes)
    make_summary(f" - Nodes TP rate: {nodes_tpr}")

  if len(false_positive_nodes | true_negative_nodes) == 0:
    nodes_fpr = "N/A"
    make_summary(f" - Nodes FP rate: N/A")
  else:
    nodes_fpr = len(false_positive_nodes) / len(false_positive_nodes | true_negative_nodes)
    make_summary(f" - Nodes FP rate: {nodes_fpr}")

  if len(true_positive_edges | false_negative_edges) == 0:
    edges_tpr = "N/A"
    make_summary(f" - Edges TP rate: N/A")
  else:
    edges_tpr = len(true_positive_edges) / len(true_positive_edges | false_negative_edges)
    make_summary(f" - Edges TP rate: {edges_tpr}")

  if len(false_positive_edges | true_negative_edges) == 0:
    edges_fpr = "N/A"
    make_summary(f" - Edges FP rate: N/A")
  else:
    edges_fpr = len(false_positive_edges) / len(false_positive_edges | true_negative_edges)
    make_summary(f" - Edges FP rate: {edges_fpr}")

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

def evaluate_hypothesis_circuit(
    hypothesis_circuit: Circuit,
    ll_model: HookedTracrTransformer,
    hl_ll_corr: TracrCorrespondence,
    case: BenchmarkCase,
    full_circuit: Optional[Circuit] = None,
    **kwargs,
):
  if full_circuit is None:
    full_corr = TLACDCCorrespondence.setup_from_model(
      ll_model, use_pos_embed=True
    )
    full_circuit = build_from_acdc_correspondence(full_corr)

  gt_circuit = get_gt_circuit(hl_ll_corr, full_circuit, ll_model.cfg.n_heads, case)

  return calculate_fpr_and_tpr(
    hypothesis_circuit, gt_circuit, full_circuit, **kwargs
  )


def build_from_acdc_correspondence(corr: TLACDCCorrespondence) -> Circuit:
    """Return a Circuit object (ACDC level granularity) from a TLACDCCorrespondence object."""
    circuit = Circuit()

    for (child_name, child_index, parent_name, parent_index), edge in corr.edge_dict().items():
        if edge.present and edge.edge_type != EdgeType.PLACEHOLDER:
            parent_head_index = None
            if (
                parent_index is not None
                and len(parent_index.hashable_tuple) > 2
                and parent_index.hashable_tuple[2] is not None
            ):
                parent_head_index = parent_index.hashable_tuple[2]

            child_head_index = None
            if (
                child_index is not None
                and len(child_index.hashable_tuple) > 2
                and child_index.hashable_tuple[2] is not None
            ):
                child_head_index = child_index.hashable_tuple[2]

            from_node = CircuitNode(parent_name, parent_head_index)
            to_node = CircuitNode(child_name, child_head_index)
            circuit.add_edge(from_node, to_node)

    return circuit


def get_full_circuit(n_layers: int, n_heads: int) -> Circuit:
    """Return a full circuit (ACDC level granularity) with n_layers and n_heads."""
    circuit = Circuit()

    circuit.add_node(CircuitNode("hook_embed"))
    circuit.add_node(CircuitNode("hook_pos_embed"))

    # nodes that write to residual stream match at least one of the following
    resid_writers_filter = ["hook_embed", "hook_pos_embed", "attn.hook_result", "hook_mlp_out"]

    for layer in range(n_layers):
        upstream_nodes = list(circuit.nodes)

        # attention heads
        for head in range(n_heads):
            for letter in "qkv":
                input_node = CircuitNode(f"blocks.{layer}.hook_{letter}_input", head)
                matrix_node = CircuitNode(f"blocks.{layer}.attn.hook_{letter}", head)
                circuit.add_node(input_node)
                circuit.add_node(matrix_node)
                circuit.add_edge(input_node, matrix_node)

            attn_result_node = CircuitNode(f"blocks.{layer}.attn.hook_result", head)
            circuit.add_node(attn_result_node)
            for letter in "qkv":
                matrix_node = CircuitNode(f"blocks.{layer}.attn.hook_{letter}", head)
                circuit.add_edge(matrix_node, attn_result_node)

            nodes_that_receive_resid_directly = [
                f"blocks.{layer}.hook_q_input",
                f"blocks.{layer}.hook_k_input",
                f"blocks.{layer}.hook_v_input",
            ]
            for from_node in upstream_nodes:
                if any([from_node.name.endswith(resid_writer) for resid_writer in resid_writers_filter]):
                    if f"blocks.{layer}" not in from_node.name:  # discard nodes from current layer
                        for to_node_name in nodes_that_receive_resid_directly:
                            circuit.add_edge(from_node, CircuitNode(to_node_name, head))

        upstream_nodes = list(circuit.nodes)

        # mlp
        mlp_in_node = CircuitNode(f"blocks.{layer}.hook_mlp_in")
        mlp_out_node = CircuitNode(f"blocks.{layer}.hook_mlp_out")
        circuit.add_node(mlp_in_node)
        circuit.add_node(mlp_out_node)
        circuit.add_edge(mlp_in_node, mlp_out_node)

        nodes_that_receive_resid_directly = [mlp_in_node]
        for from_node in upstream_nodes:
            if any([from_node.name.endswith(resid_writer) for resid_writer in resid_writers_filter]):
                if (f"blocks.{layer}" not in from_node.name) or ("attn.hook_result" in from_node.name):
                    for to_node in nodes_that_receive_resid_directly:
                        circuit.add_edge(from_node, to_node)

    last_resid_post_node = CircuitNode(f"blocks.{n_layers - 1}.hook_resid_post")
    circuit.add_node(last_resid_post_node)
    for from_node in list(circuit.nodes):
        if any([from_node.name.endswith(resid_writer) for resid_writer in resid_writers_filter]):
            circuit.add_edge(from_node, last_resid_post_node)

    return circuit


def replace_inputs_and_qkv_nodes_with_outputs(circuit: Circuit) -> Circuit:
    new_circuit = Circuit()
    prefix = lambda node_name: ".".join(node_name.split(".")[:-1])
    suffix = lambda node_name: node_name.split(".")[-1]

    for node in circuit.nodes:
        node_name_prefix = prefix(node.name)
        node_name_suffix = suffix(node.name)
        if node_name_suffix in ["hook_q_input", "hook_k_input", "hook_v_input"]:
            new_node = CircuitNode(f"{node_name_prefix}.attn.hook_result", node.index)
        elif node_name_suffix in ["hook_embed", "hook_pos_embed", "hook_resid_pre"]:
            continue
        elif node_name_suffix == "hook_mlp_in":
            # TODO:
            # We are doing this because SP removes this type of direct computation edge
            # this needs to be fixed in the SP code
            new_node = CircuitNode(f"{node_name_prefix}.hook_mlp_out")
        elif node_name_suffix in ["hook_q", "hook_k", "hook_v"]:
            new_node = CircuitNode(f"{node_name_prefix}.hook_result", node.index)
        else:
            new_node = node
        new_circuit.add_node(new_node)

    return new_circuit.nodes


def replace_inputs_and_qkv_edges_with_outputs(circuit: Circuit) -> Circuit:
    new_circuit = Circuit()
    prefix = lambda node_name: ".".join(node_name.split(".")[:-1])
    suffix = lambda node_name: node_name.split(".")[-1]

    for from_node, to_node in circuit.edges:
        from_node_name_prefix = prefix(from_node.name)
        from_node_name_suffix = suffix(from_node.name)
        to_node_name_prefix = prefix(to_node.name)
        to_node_name_suffix = suffix(to_node.name)
        qkv_ins = ["hook_q_input", "hook_k_input", "hook_v_input"]
        qkv_outs = ["hook_q", "hook_k", "hook_v"]
        resids = ["hook_resid_post", "hook_resid_pre"] # TODO: maybe not the last one

        if (
           (from_node_name_suffix in qkv_ins)  # Placeholder: {qkv}_input -> hook_{qkv}
        or (from_node_name_suffix in qkv_outs)  # Direct computation: hook_q -> hook_result
        or (from_node_name_suffix == "hook_mlp_in") # Direct computation: hook_mlp_in -> hook_mlp_out
        ):
            # Ignore direct computation and placeholder edges
            if not (
                (to_node_name_suffix in qkv_outs)
                or (to_node_name_suffix == "hook_result")
                or (to_node_name_suffix == "hook_mlp_out")
            ):
                print(
                    f"!!! WARNING: Received an invalid edge:",
                    f"{from_node.name} -> {to_node.name}"
                )
            continue
        elif from_node_name_suffix in resids or to_node_name_suffix in resids:
            # ignore resid edges: TODO: Not sure if I should consider these edges...
            continue
        elif to_node_name_suffix in qkv_ins:
            # directly route incoming edges to head's hook_result
            new_from_node = from_node
            new_to_node = CircuitNode(f"{to_node_name_prefix}.attn.hook_result", to_node.index)
        elif to_node_name_suffix == "hook_mlp_in":
            # directly route incoming edges to mlp_out
            new_from_node = from_node
            new_to_node = CircuitNode(f"{to_node_name_prefix}.hook_mlp_out")
        else:
            # all other edges are okay...
            new_from_node = from_node
            new_to_node = to_node
        new_circuit.add_edge(new_from_node, new_to_node)
    return new_circuit.edges
