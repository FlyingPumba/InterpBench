from typing import Optional

import torch
import torch.nn.functional as F


def kl_divergence(
    logits: torch.Tensor,
    base_model_logprobs: torch.Tensor,
    mask_repeat_candidates: Optional[torch.Tensor] = None,
    last_seq_element_only: bool = True,
    base_model_probs_last_seq_element_only: bool = False,
    return_one_element: bool = True,
) -> torch.Tensor:
  # Note: we want base_model_probs_last_seq_element_only to remain False by default, because when the Docstring
  # circuit uses this, it already takes the last position before passing it in.

  if last_seq_element_only:
    logits = logits[:, -1, :]

  if base_model_probs_last_seq_element_only:
    base_model_logprobs = base_model_logprobs[:, -1, :]

  logprobs = F.log_softmax(logits, dim=-1)
  kl_div = F.kl_div(logprobs, base_model_logprobs, log_target=True, reduction="none").sum(dim=-1)

  if mask_repeat_candidates is not None:
    assert kl_div.shape == mask_repeat_candidates.shape, (kl_div.shape, mask_repeat_candidates.shape)
    answer = kl_div[mask_repeat_candidates]
  elif not last_seq_element_only:
    assert kl_div.ndim == 2, kl_div.shape
    answer = kl_div.view(-1)
  else:
    answer = kl_div

  if return_one_element:
    return answer.mean()

  return answer


def l2_metric(new_output: torch.Tensor,
              baseline_output: torch.Tensor,
              is_categorical: bool = True,
              discard_bos_token: bool = True):
  assert new_output.shape == baseline_output.shape, (new_output.shape, baseline_output.shape)

  if discard_bos_token:
    new_output = new_output[:, 1:]
    baseline_output = baseline_output[:, 1:]

  if not is_categorical:
    # then the output is numerical, and we retain only the output for the first logit.
    new_output = new_output[:, :, 0]
    baseline_output = baseline_output[:, :, 0]

  return ((new_output - baseline_output) ** 2).mean()