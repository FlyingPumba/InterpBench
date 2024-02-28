import unittest

from circuits_benchmark.benchmark.cases.case_3 import Case3


class CircuitTest(unittest.TestCase):
  def test_build_circuit_for_case_3(self):
    case = Case3()
    circuit = case.get_tracr_circuit()

    expected_nodes = ["embed.W_E", "pos_embed.W_pos", "blocks.0.mlp", "blocks.1.attn"]
    expected_edges = [("embed.W_E", "blocks.0.mlp"),
                      ("pos_embed.W_pos", "blocks.1.attn"),
                      ("blocks.0.mlp", "blocks.1.attn")]
    self.assertEqual(sorted(circuit.nodes), sorted(expected_nodes))
    self.assertEqual(sorted(circuit.edges), sorted(expected_edges))
