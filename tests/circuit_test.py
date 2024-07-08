import unittest

from circuits_benchmark.benchmark.cases.case_16 import Case16
from circuits_benchmark.benchmark.cases.case_3 import Case3
from circuits_benchmark.transformers.tracr_circuits_builder import build_tracr_circuits


class CircuitTest(unittest.TestCase):
  def test_build_circuit_for_case_3_with_component_granularity(self):
    case = Case3()
    tracr_transformer_circuit = case.get_hl_gt_circuit(granularity="component")

    expected_nodes = ["embed", "pos_embed", "blocks.0.mlp", "blocks.1.attn[0]"]
    expected_edges = [("embed", "blocks.0.mlp"),
                      ("pos_embed", "blocks.1.attn[0]"),
                      ("blocks.0.mlp", "blocks.1.attn[0]")]
    self.assertEqual(sorted([str(n) for n in tracr_transformer_circuit.nodes]), sorted(expected_nodes))
    self.assertEqual(sorted([(str(u), str(v)) for u, v in tracr_transformer_circuit.edges]), sorted(expected_edges))

  def test_build_circuit_for_case_3_with_matrix_granularity(self):
    case = Case3()
    tracr_transformer_circuit = case.get_hl_gt_circuit(granularity="matrix")

    expected_nodes = ["embed.W_E", "pos_embed.W_pos",
                      "blocks.0.mlp.W_in", "blocks.0.mlp.W_out",
                      "blocks.1.attn.W_Q[0]", "blocks.1.attn.W_K[0]",
                      "blocks.1.attn.W_V[0]", "blocks.1.attn.W_O[0]"]
    expected_edges = [("embed.W_E", "blocks.0.mlp.W_in"),
                      ("pos_embed.W_pos", "blocks.1.attn.W_K[0]"),
                      ("pos_embed.W_pos", "blocks.1.attn.W_Q[0]"),
                      ("blocks.0.mlp.W_in", "blocks.0.mlp.W_out"),
                      ("blocks.0.mlp.W_out", "blocks.1.attn.W_V[0]"),
                      ("blocks.1.attn.W_K[0]", "blocks.1.attn.W_O[0]"),
                      ("blocks.1.attn.W_Q[0]", "blocks.1.attn.W_O[0]"),
                      ("blocks.1.attn.W_V[0]", "blocks.1.attn.W_O[0]")]
    self.assertEqual(sorted([str(n) for n in tracr_transformer_circuit.nodes]), sorted(expected_nodes))
    self.assertEqual(sorted([(str(u), str(v)) for u, v in tracr_transformer_circuit.edges]), sorted(expected_edges))

  def test_build_circuit_for_case_3_with_acdc_hooks_granularity(self):
    case = Case3()
    tracr_transformer_circuit = case.get_hl_gt_circuit(granularity="acdc_hooks")

    expected_nodes = ["hook_embed", "hook_pos_embed",
                      "blocks.0.hook_mlp_in", "blocks.0.hook_mlp_out",
                      "blocks.1.hook_q_input[0]", "blocks.1.hook_k_input[0]", "blocks.1.hook_v_input[0]",
                      "blocks.1.attn.hook_q[0]", "blocks.1.attn.hook_k[0]", "blocks.1.attn.hook_v[0]",
                      "blocks.1.attn.hook_result[0]", "blocks.1.hook_resid_post"]
    expected_edges = [("hook_embed", "blocks.0.hook_mlp_in"),
                      ("hook_pos_embed", "blocks.1.hook_q_input[0]"),
                      ("hook_pos_embed", "blocks.1.hook_k_input[0]"),
                      ("blocks.0.hook_mlp_in", "blocks.0.hook_mlp_out"),
                      ("blocks.0.hook_mlp_out", "blocks.1.hook_v_input[0]"),
                      ("blocks.1.hook_q_input[0]", "blocks.1.attn.hook_q[0]"),
                      ("blocks.1.hook_k_input[0]", "blocks.1.attn.hook_k[0]"),
                      ("blocks.1.hook_v_input[0]", "blocks.1.attn.hook_v[0]"),
                      ("blocks.1.attn.hook_q[0]", "blocks.1.attn.hook_result[0]"),
                      ("blocks.1.attn.hook_k[0]", "blocks.1.attn.hook_result[0]"),
                      ("blocks.1.attn.hook_v[0]", "blocks.1.attn.hook_result[0]"),
                      ("blocks.1.attn.hook_result[0]", "blocks.1.hook_resid_post")]
    self.assertEqual(sorted([str(n) for n in tracr_transformer_circuit.nodes]), sorted(expected_nodes))
    self.assertEqual(sorted([(str(u), str(v)) for u, v in tracr_transformer_circuit.edges]), sorted(expected_edges))

  def test_build_circuit_for_case_3_with_sp_hooks_granularity(self):
    case = Case3()
    tracr_transformer_circuit = case.get_hl_gt_circuit(granularity="sp_hooks")

    expected_nodes = ["hook_embed", "hook_pos_embed",
                      "blocks.0.hook_mlp_in", "blocks.0.hook_mlp_out",
                      "blocks.1.hook_q_input[0]", "blocks.1.hook_k_input[0]", "blocks.1.hook_v_input[0]",
                      "blocks.1.attn.hook_result[0]", "blocks.1.hook_resid_post"]
    expected_edges = [("hook_embed", "blocks.0.hook_mlp_in"),
                      ("hook_pos_embed", "blocks.1.hook_q_input[0]"),
                      ("hook_pos_embed", "blocks.1.hook_k_input[0]"),
                      ("blocks.0.hook_mlp_in", "blocks.0.hook_mlp_out"),
                      ("blocks.0.hook_mlp_out", "blocks.1.hook_v_input[0]"),
                      ("blocks.1.hook_q_input[0]", "blocks.1.attn.hook_result[0]"),
                      ("blocks.1.hook_k_input[0]", "blocks.1.attn.hook_result[0]"),
                      ("blocks.1.hook_v_input[0]", "blocks.1.attn.hook_result[0]"),
                      ("blocks.1.attn.hook_result[0]", "blocks.1.hook_resid_post")]
    self.assertEqual(sorted([str(n) for n in tracr_transformer_circuit.nodes]), sorted(expected_nodes))
    self.assertEqual(sorted([(str(u), str(v)) for u, v in tracr_transformer_circuit.edges]), sorted(expected_edges))

  def test_build_circuit_for_case_16_with_acdc_hooks_granularity(self):
    case = Case16()
    tracr_transformer_circuit = case.get_hl_gt_circuit(granularity="acdc_hooks")

    expected_nodes = ["blocks.0.attn.hook_k[0]", "blocks.0.attn.hook_q[0]", "blocks.0.attn.hook_result[0]",
                      "blocks.0.attn.hook_v[0]", "blocks.0.hook_k_input[0]", "blocks.0.hook_mlp_in",
                      "blocks.0.hook_mlp_out", "blocks.0.hook_q_input[0]", "blocks.0.hook_v_input[0]",
                      "blocks.1.attn.hook_k[0]", "blocks.1.attn.hook_q[0]", "blocks.1.attn.hook_result[0]",
                      "blocks.1.attn.hook_v[0]", "blocks.1.hook_k_input[0]", "blocks.1.hook_mlp_in",
                      "blocks.1.hook_mlp_out", "blocks.1.hook_q_input[0]", "blocks.1.hook_v_input[0]",
                      "blocks.2.hook_mlp_in", "blocks.2.hook_mlp_out", "blocks.3.attn.hook_k[0]",
                      "blocks.3.attn.hook_q[0]", "blocks.3.attn.hook_result[0]", "blocks.3.attn.hook_v[0]",
                      "blocks.3.hook_k_input[0]", "blocks.3.hook_q_input[0]", "blocks.3.hook_v_input[0]",
                      "blocks.4.attn.hook_k[0]", "blocks.4.attn.hook_q[0]", "blocks.4.attn.hook_result[0]",
                      "blocks.4.attn.hook_v[0]", "blocks.4.hook_k_input[0]", "blocks.4.hook_mlp_in",
                      "blocks.4.hook_mlp_out", "blocks.4.hook_q_input[0]", "blocks.4.hook_v_input[0]",
                      "blocks.5.hook_mlp_in", "blocks.5.hook_mlp_out", "blocks.6.hook_mlp_in",
                      "blocks.6.hook_mlp_out", "blocks.6.hook_resid_post", "hook_embed", "hook_pos_embed"]
    expected_edges = [("blocks.0.attn.hook_k[0]", "blocks.0.attn.hook_result[0]"),
                      ("blocks.0.attn.hook_k[0]", "blocks.3.hook_v_input[0]"),
                      ("blocks.0.attn.hook_q[0]", "blocks.0.attn.hook_result[0]"),
                      ("blocks.0.attn.hook_q[0]", "blocks.3.hook_v_input[0]"),
                      ("blocks.0.attn.hook_result[0]", "blocks.0.hook_mlp_in"),
                      ("blocks.0.attn.hook_v[0]", "blocks.0.attn.hook_result[0]"),
                      ("blocks.0.hook_k_input[0]", "blocks.0.attn.hook_k[0]"),
                      ("blocks.0.hook_mlp_in", "blocks.0.hook_mlp_out"),
                      ("blocks.0.hook_mlp_out", "blocks.1.hook_k_input[0]"),
                      ("blocks.0.hook_mlp_out", "blocks.1.hook_q_input[0]"),
                      ("blocks.0.hook_mlp_out", "blocks.5.hook_mlp_in"),
                      ("blocks.0.hook_q_input[0]", "blocks.0.attn.hook_q[0]"),
                      ("blocks.0.hook_v_input[0]", "blocks.0.attn.hook_v[0]"),
                      ("blocks.1.attn.hook_k[0]", "blocks.1.attn.hook_result[0]"),
                      ("blocks.1.attn.hook_q[0]", "blocks.1.attn.hook_result[0]"),
                      ("blocks.1.attn.hook_result[0]", "blocks.1.hook_mlp_in"),
                      ("blocks.1.attn.hook_v[0]", "blocks.1.attn.hook_result[0]"),
                      ("blocks.1.hook_k_input[0]", "blocks.1.attn.hook_k[0]"),
                      ("blocks.1.hook_mlp_in", "blocks.1.hook_mlp_out"),
                      ("blocks.1.hook_mlp_out", "blocks.2.hook_mlp_in"),
                      ("blocks.1.hook_q_input[0]", "blocks.1.attn.hook_q[0]"),
                      ("blocks.1.hook_v_input[0]", "blocks.1.attn.hook_v[0]"),
                      ("blocks.2.hook_mlp_in", "blocks.2.hook_mlp_out"),
                      ("blocks.2.hook_mlp_out", "blocks.3.hook_k_input[0]"),
                      ("blocks.3.attn.hook_k[0]", "blocks.3.attn.hook_result[0]"),
                      ("blocks.3.attn.hook_q[0]", "blocks.3.attn.hook_result[0]"),
                      ("blocks.3.attn.hook_result[0]", "blocks.4.hook_k_input[0]"),
                      ("blocks.3.attn.hook_result[0]", "blocks.4.hook_q_input[0]"),
                      ("blocks.3.attn.hook_v[0]", "blocks.3.attn.hook_result[0]"),
                      ("blocks.3.hook_k_input[0]", "blocks.3.attn.hook_k[0]"),
                      ("blocks.3.hook_q_input[0]", "blocks.3.attn.hook_q[0]"),
                      ("blocks.3.hook_v_input[0]", "blocks.3.attn.hook_v[0]"),
                      ("blocks.4.attn.hook_k[0]", "blocks.4.attn.hook_result[0]"),
                      ("blocks.4.attn.hook_q[0]", "blocks.4.attn.hook_result[0]"),
                      ("blocks.4.attn.hook_result[0]", "blocks.4.hook_mlp_in"),
                      ("blocks.4.attn.hook_v[0]", "blocks.4.attn.hook_result[0]"),
                      ("blocks.4.hook_k_input[0]", "blocks.4.attn.hook_k[0]"),
                      ("blocks.4.hook_mlp_in", "blocks.4.hook_mlp_out"),
                      ("blocks.4.hook_mlp_out", "blocks.5.hook_mlp_in"),
                      ("blocks.4.hook_q_input[0]", "blocks.4.attn.hook_q[0]"),
                      ("blocks.4.hook_v_input[0]", "blocks.4.attn.hook_v[0]"),
                      ("blocks.5.hook_mlp_in", "blocks.5.hook_mlp_out"),
                      ("blocks.5.hook_mlp_out", "blocks.6.hook_mlp_in"),
                      ("blocks.6.hook_mlp_in", "blocks.6.hook_mlp_out"),
                      ("blocks.6.hook_mlp_out", "blocks.6.hook_resid_post"),
                      ("hook_embed", "blocks.0.hook_k_input[0]"),
                      ("hook_embed", "blocks.0.hook_mlp_in"),
                      ("hook_embed", "blocks.0.hook_q_input[0]"),
                      ("hook_embed", "blocks.3.hook_v_input[0]"),
                      ("hook_pos_embed", "blocks.0.hook_mlp_in"),
                      ("hook_pos_embed", "blocks.2.hook_mlp_in"),
                      ("hook_pos_embed", "blocks.3.hook_q_input[0]")]
    self.assertEqual(sorted([str(n) for n in tracr_transformer_circuit.nodes]), sorted(expected_nodes))
    self.assertEqual(sorted([(str(u), str(v)) for u, v in tracr_transformer_circuit.edges]), sorted(expected_edges))

  def test_all_keys_are_strings_in_hl_to_ll_mapping(self):
    case = Case3()
    tacr_output = case.get_tracr_output()
    tracr_circuits = build_tracr_circuits(tacr_output.graph, tacr_output.craft_model, granularity="acdc_hooks")
    for k, v in tracr_circuits.alignment.hl_to_ll_mapping.items():
      self.assertIsInstance(k, str)
