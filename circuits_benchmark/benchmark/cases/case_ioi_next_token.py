from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule

from circuits_benchmark.benchmark.cases.case_ioi import CaseIOI
from iit.model_pairs.base_model_pair import BaseModelPair
from iit.model_pairs.ioi_model_pair import IOI_ModelPair
from iit.utils.correspondence import Correspondence


class CaseIOI_Next_Token(CaseIOI):
  def get_task_description(self) -> str:
    """Returns the task description for the benchmark case."""
    return "Indirect Object Identification (IOI) task, trained using next token prediction."

  def build_model_pair(
      self,
      training_args: dict | None = None,
      ll_model: HookedTransformer | None = None,
      hl_model: HookedRootModule | None = None,
      hl_ll_corr: Correspondence | None = None,
      *args, **kwargs
  ) -> BaseModelPair:
    if training_args is None:
      training_args = {}

    if ll_model is None:
      ll_model = self.get_ll_model()

    if hl_model is None:
      hl_model = self.get_hl_model()

    if hl_ll_corr is None:
      hl_ll_corr = self.get_correspondence()

    training_args["next_token"] = True

    return IOI_ModelPair(
      ll_model=ll_model,
      hl_model=hl_model,
      corr=hl_ll_corr,
      training_args=training_args,
    )