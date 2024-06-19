from typing import Optional

from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.transformers.acdc_circuit_builder import build_acdc_circuit
from circuits_benchmark.transformers.circuit import Circuit
from circuits_benchmark.transformers.hooked_tracr_transformer import HookedTracrTransformer
from circuits_benchmark.utils.circuits_comparison import calculate_fpr_and_tpr
from circuits_benchmark.utils.iit._acdc_utils import get_gt_circuit
from circuits_benchmark.utils.iit.correspondence import TracrCorrespondence


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
    full_circuit = build_acdc_circuit(corr=full_corr)

  gt_circuit = get_gt_circuit(hl_ll_corr, full_circuit, ll_model.cfg.n_heads, case)

  return calculate_fpr_and_tpr(
    hypothesis_circuit, gt_circuit, full_circuit, **kwargs
  )
