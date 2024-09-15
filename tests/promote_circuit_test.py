from circuits_benchmark.utils.circuit.edges_list import edges_list_to_circuit, circuit_to_edges_list
from circuits_benchmark.utils.circuit.prepare_circuit import prepare_circuit_for_evaluation
from circuits_benchmark.utils.project_paths import detect_project_root


class TestCircuitPromotion:
    def test_promotion_removes_attn_inputs(self):
        orig_edges = [('blocks.1.attn.hook_result[2]', 'blocks.1.hook_resid_post'),
                      ('blocks.0.hook_mlp_out', 'blocks.1.hook_q_input[2]'),
                      ('blocks.0.hook_mlp_out', 'blocks.1.hook_k_input[2]'),
                      ('blocks.0.hook_mlp_out', 'blocks.1.hook_v_input[2]'),
                      ('hook_embed', 'blocks.0.hook_mlp_in'),
                      ('hook_pos_embed', 'blocks.1.hook_q_input[2]'),
                      ('hook_pos_embed', 'blocks.1.hook_k_input[2]'),
                      ('hook_pos_embed', 'blocks.1.hook_v_input[2]'),
                      ('blocks.1.hook_q_input[2]', 'blocks.1.attn.hook_q[2]'),
                      ('blocks.1.hook_k_input[2]', 'blocks.1.attn.hook_k[2]'),
                      ('blocks.1.hook_v_input[2]', 'blocks.1.attn.hook_v[2]')]

        orig_circuit = edges_list_to_circuit(orig_edges)
        promoted_circuit = prepare_circuit_for_evaluation(orig_circuit)

        expected_new_edges = [('blocks.1.attn.hook_result[2]', 'blocks.1.hook_resid_post'),
                              ('blocks.0.hook_mlp_out', 'blocks.1.attn.hook_result[2]'),
                              ('blocks.0.hook_resid_pre', 'blocks.0.hook_mlp_out'),
                              ('blocks.0.hook_resid_pre', 'blocks.1.attn.hook_result[2]')]

        assert expected_new_edges == circuit_to_edges_list(promoted_circuit)

    def test_removes_direct_computation_and_placeholder_edges(self):
        orig_edges = [('blocks.0.hook_resid_pre', 'blocks.0.hook_mlp_in'),
                      ('blocks.0.hook_resid_pre', 'blocks.1.hook_resid_post'),
                      ('blocks.0.hook_resid_pre', 'blocks.1.hook_v_input[3]'),
                      ('blocks.0.hook_resid_pre', 'blocks.1.hook_k_input[3]'),
                      ('blocks.0.hook_resid_pre', 'blocks.1.hook_v_input[2]'),
                      ('blocks.0.hook_resid_pre', 'blocks.1.hook_v_input[0]'),
                      ('blocks.0.attn.hook_result[0]', 'blocks.1.hook_v_input[3]'),
                      ('blocks.0.attn.hook_result[0]', 'blocks.1.hook_v_input[2]'),
                      ('blocks.0.attn.hook_result[0]', 'blocks.0.hook_mlp_in'),
                      ('blocks.1.attn.hook_result[3]', 'blocks.1.hook_resid_post'),
                      ('blocks.1.attn.hook_result[0]', 'blocks.1.hook_resid_post'),
                      ('blocks.0.hook_mlp_out', 'blocks.1.hook_resid_post'),
                      ('blocks.0.hook_mlp_out', 'blocks.1.hook_mlp_in'),
                      ('blocks.0.hook_mlp_out', 'blocks.1.hook_v_input[3]'),
                      ('blocks.0.hook_mlp_out', 'blocks.1.hook_v_input[2]'),
                      ('blocks.0.hook_mlp_out', 'blocks.1.hook_v_input[0]'),
                      ('blocks.0.hook_mlp_out', 'blocks.1.hook_k_input[3]'),
                      ('blocks.1.attn.hook_result[2]', 'blocks.1.hook_resid_post'),
                      ('blocks.1.attn.hook_result[2]', 'blocks.1.hook_mlp_in'),
                      ('blocks.1.hook_mlp_out', 'blocks.1.hook_resid_post'),
                      ('blocks.0.attn.hook_result[1]', 'blocks.1.hook_v_input[2]')]

        orig_circuit = edges_list_to_circuit(orig_edges)
        promoted_circuit = prepare_circuit_for_evaluation(orig_circuit)

        expected_new_edges = [('blocks.0.hook_resid_pre', 'blocks.0.hook_mlp_out'),
                              ('blocks.0.hook_resid_pre', 'blocks.1.attn.hook_result[3]'),
                              ('blocks.0.hook_resid_pre', 'blocks.1.attn.hook_result[2]'),
                              ('blocks.0.hook_resid_pre', 'blocks.1.attn.hook_result[0]'),
                              ('blocks.0.hook_mlp_out', 'blocks.1.hook_resid_post'),
                              ('blocks.0.hook_mlp_out', 'blocks.1.hook_mlp_out'),
                              ('blocks.0.hook_mlp_out', 'blocks.1.attn.hook_result[3]'),
                              ('blocks.0.hook_mlp_out', 'blocks.1.attn.hook_result[2]'),
                              ('blocks.0.hook_mlp_out', 'blocks.1.attn.hook_result[0]'),
                              ('blocks.1.attn.hook_result[3]', 'blocks.1.hook_resid_post'),
                              ('blocks.1.attn.hook_result[2]', 'blocks.1.hook_resid_post'),
                              ('blocks.1.attn.hook_result[2]', 'blocks.1.hook_mlp_out'),
                              ('blocks.1.attn.hook_result[0]', 'blocks.1.hook_resid_post'),
                              ('blocks.0.attn.hook_result[0]', 'blocks.1.attn.hook_result[3]'),
                              ('blocks.0.attn.hook_result[0]', 'blocks.1.attn.hook_result[2]'),
                              ('blocks.0.attn.hook_result[0]', 'blocks.0.hook_mlp_out'),
                              ('blocks.1.hook_mlp_out', 'blocks.1.hook_resid_post'),
                              ('blocks.0.attn.hook_result[1]', 'blocks.1.attn.hook_result[2]')]

        assert expected_new_edges == circuit_to_edges_list(promoted_circuit)

    def test_no_removal_or_rerouting(self):
        """
        Test case where no edges are removed or rerouted.
        """
        orig_edges = [
            ('blocks.0.hook_resid_pre', 'blocks.0.hook_mlp_in'),
            ('blocks.0.hook_mlp_out', 'blocks.1.hook_resid_post'),
            ('blocks.0.hook_mlp_out', 'blocks.1.hook_q_input[2]'),
            ('hook_embed', 'blocks.0.hook_resid_pre'),
            ('blocks.1.hook_q_input[2]', 'blocks.1.attn.hook_q[2]')
        ]

        orig_circuit = edges_list_to_circuit(orig_edges)
        promoted_circuit = prepare_circuit_for_evaluation(
            orig_circuit,
            remove_edges_from_qkv_inputs=False,
            reroute_edges_to_qkv_inputs=False,
            remove_edges_from_mlp_in=False,
            reroute_edges_to_mlp_in=False,
            rename_edges_from_embed_to_resid_pre=False,
            remove_embed_to_resid_edges=False,
            remove_ignorable_resid_edges=False
        )

        # Expect the same edges because no removal or rerouting is performed
        assert set(orig_edges) == set(circuit_to_edges_list(promoted_circuit))

    def test_only_reroute_qkv_inputs(self):
        """
        Test case where edges are only rerouted to QKV inputs.
        """
        orig_edges = [
            ('blocks.0.hook_mlp_out', 'blocks.1.hook_q_input[2]'),
            ('blocks.0.hook_mlp_out', 'blocks.1.hook_k_input[2]'),
            ('blocks.0.hook_mlp_out', 'blocks.1.hook_v_input[2]')
        ]

        orig_circuit = edges_list_to_circuit(orig_edges)
        promoted_circuit = prepare_circuit_for_evaluation(
            orig_circuit,
            remove_edges_from_qkv_inputs=False,
            reroute_edges_to_qkv_inputs=True,
            remove_edges_from_mlp_in=False,
            reroute_edges_to_mlp_in=False,
            rename_edges_from_embed_to_resid_pre=False,
            remove_embed_to_resid_edges=False,
            remove_ignorable_resid_edges=False
        )

        expected_new_edges = [
            ('blocks.0.hook_mlp_out', 'blocks.1.attn.hook_result[2]'),
        ]

        assert expected_new_edges == circuit_to_edges_list(promoted_circuit)

    def test_remove_embed_to_resid_edges(self):
        """
        Test case where edges from embed nodes to resid nodes are removed.
        """
        orig_edges = [
            ('hook_embed', 'blocks.0.hook_resid_pre'),
            ('hook_pos_embed', 'blocks.0.hook_resid_pre'),
            ('blocks.0.hook_resid_pre', 'blocks.0.hook_mlp_out'),
            ('blocks.0.hook_mlp_out', 'blocks.1.hook_resid_post')
        ]

        orig_circuit = edges_list_to_circuit(orig_edges)
        promoted_circuit = prepare_circuit_for_evaluation(
            orig_circuit,
            remove_edges_from_qkv_inputs=False,
            reroute_edges_to_qkv_inputs=False,
            remove_edges_from_mlp_in=False,
            reroute_edges_to_mlp_in=False,
            rename_edges_from_embed_to_resid_pre=False,
            remove_embed_to_resid_edges=True,
            remove_ignorable_resid_edges=False
        )

        expected_new_edges = [
            ('blocks.0.hook_resid_pre', 'blocks.0.hook_mlp_out'),
            ('blocks.0.hook_mlp_out', 'blocks.1.hook_resid_post')
        ]

        assert expected_new_edges == circuit_to_edges_list(promoted_circuit)

    def test_rename_embed_to_resid_pre(self):
        """
        Test case where edges from embed nodes are renamed to hook_resid_pre.
        """
        orig_edges = [
            ('hook_embed', 'blocks.0.hook_mlp_in'),
            ('hook_pos_embed', 'blocks.1.hook_resid_post')
        ]

        orig_circuit = edges_list_to_circuit(orig_edges)
        promoted_circuit = prepare_circuit_for_evaluation(
            orig_circuit,
            remove_edges_from_qkv_inputs=False,
            reroute_edges_to_qkv_inputs=False,
            remove_edges_from_mlp_in=False,
            reroute_edges_to_mlp_in=False,
            rename_edges_from_embed_to_resid_pre=True,
            remove_embed_to_resid_edges=False,
            remove_ignorable_resid_edges=False
        )

        expected_new_edges = [
            ('blocks.0.hook_resid_pre', 'blocks.0.hook_mlp_in'),
            ('blocks.0.hook_resid_pre', 'blocks.1.hook_resid_post')
        ]

        assert expected_new_edges == circuit_to_edges_list(promoted_circuit)

    def test_reroute_mlp_inputs(self):
        """
        Test case where edges to mlp_in nodes are rerouted to mlp_out.
        """
        orig_edges = [
            ('blocks.0.hook_resid_pre', 'blocks.1.hook_mlp_in')
        ]

        orig_circuit = edges_list_to_circuit(orig_edges)
        promoted_circuit = prepare_circuit_for_evaluation(
            orig_circuit,
            remove_edges_from_qkv_inputs=False,
            reroute_edges_to_qkv_inputs=False,
            remove_edges_from_mlp_in=False,
            reroute_edges_to_mlp_in=True,
            rename_edges_from_embed_to_resid_pre=False,
            remove_embed_to_resid_edges=False,
            remove_ignorable_resid_edges=False
        )

        expected_new_edges = [
            ('blocks.0.hook_resid_pre', 'blocks.1.hook_mlp_out')
        ]

        assert expected_new_edges == circuit_to_edges_list(promoted_circuit)

    def test_many_cases(self):
        with open(detect_project_root() + "/tests/promote_circuit_test_orig_edges.txt") as f:
            orig_edges_lines = f.readlines()

        with open(detect_project_root() + "/tests/promote_circuit_test_new_edges.txt") as f:
            new_edges_lines = f.readlines()

        for orig_line, new_line in zip(orig_edges_lines, new_edges_lines):
            orig_edges = eval(orig_line)
            orig_circuit = edges_list_to_circuit(orig_edges)

            promoted_circuit = prepare_circuit_for_evaluation(orig_circuit)
            expected_new_edges = set(eval(new_line))

            assert expected_new_edges == set(circuit_to_edges_list(promoted_circuit))
