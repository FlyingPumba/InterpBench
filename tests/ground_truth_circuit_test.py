from acdc.TLACDCCorrespondence import TLACDCCorrespondence

from circuits_benchmark.benchmark.cases.case_7 import Case7
from circuits_benchmark.utils.circuit.circuit_eval import build_from_acdc_correspondence
from circuits_benchmark.utils.iit._acdc_utils import get_gt_circuit


class TestGroundTruthCircuit:
  def test_gt_circuit_for_case_7(self):
    case = Case7()

    full_corr = TLACDCCorrespondence.setup_from_model(case.get_ll_model())
    full_circuit = build_from_acdc_correspondence(corr=full_corr)

    corr = case.get_correspondence()

    gt_circuit = get_gt_circuit(
      hl_ll_corr=corr,
      full_circuit=full_circuit,
      n_heads=case.get_ll_model().cfg.n_heads,
      case=case,
    )

    assert len(gt_circuit.edges) > 0