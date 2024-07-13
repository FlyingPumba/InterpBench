from acdc.TLACDCCorrespondence import TLACDCCorrespondence

from circuits_benchmark.utils.circuit.circuit_eval import build_from_acdc_correspondence
from circuits_benchmark.utils.get_cases import get_cases
from circuits_benchmark.utils.iit._acdc_utils import get_gt_circuit
from circuits_benchmark.benchmark.cases.case_ioi import CaseIOI


class TestGroundTruthCircuit:

  def test_gt_circuit_for_all_cases(self):
    cases = get_cases()
    cases = [case for case in cases if not isinstance(case, CaseIOI)] # remove ioi cases

    for case in cases:
      full_corr = TLACDCCorrespondence.setup_from_model(case.get_ll_model())
      full_circuit = build_from_acdc_correspondence(corr=full_corr)

      corr = case.get_correspondence()

      gt_circuit = get_gt_circuit(
        hl_ll_corr=corr,
        full_circuit=full_circuit,
        n_heads=case.get_ll_model().cfg.n_heads,
        case=case,
      )

      assert len(gt_circuit.edges) > 0, f"Case {case} has no edges in gt_circuit"