from typing import Any, List

import numpy as np
import torch as t
from jaxtyping import Float, Bool
from torch import Tensor

from circuits_benchmark.benchmark.vocabs import TRACR_BOS, TRACR_PAD


def compare_positions_excluding_BOS(expected_output: List[Any],
                                    predicted_output: List[Any],
                                    is_categorical: bool,
                                    atol: float = 1.e-5) -> List[bool]:
  """Return a list of booleans indicating if the predicted output is correct at each position.
  This method only considers the positions that are not the first element.
  """
  expected_output = expected_output[1:]
  predicted_output = predicted_output[1:]
  return compare_positions(expected_output, predicted_output, is_categorical, atol)


def compare_valid_positions(expected_output: List[Any],
                            predicted_output: List[Any],
                            is_categorical: bool,
                            atol: float = 1.e-5) -> List[bool]:
  """Return a list of booleans indicating if the predicted output is correct at each position.
  This method only considers the positions that are not "BOS", "PAD", or None in the expected output.
  """
  expected_output, predicted_output = remove_invalid_positions(expected_output, predicted_output)
  return compare_positions(expected_output, predicted_output, is_categorical, atol)


def compare_positions(expected_output, predicted_output, is_categorical, atol):
  if is_categorical:
    correct_positions = [elem1 == elem2 for elem1, elem2 in zip(predicted_output, expected_output)]
  else:
    # compare how close the outputs are numerically without taking into account the BOS or PAD tokens
    correct_positions = np.isclose(expected_output, predicted_output, atol=atol).tolist()
  return correct_positions


def remove_invalid_positions(
    expected_output: List[Any],
    predicted_output: List[Any]
) -> (List[Any], List[Any]):
  """Return the expected and predicted outputs without the invalid positions."""
  assert not isinstance(expected_output[0], list), "expected_output should be a single output"

  # Figure out the indices in expected output that are "BOS", "PAD" or None
  skip_indices = set([i for i, elem in enumerate(expected_output) if elem in [TRACR_BOS, TRACR_PAD, None]])

  # Remove such elements from expected and predicted output
  expected_output = [elem for i, elem in enumerate(expected_output) if i not in skip_indices]
  predicted_output = [elem for i, elem in enumerate(predicted_output) if i not in skip_indices]

  return expected_output, predicted_output

def replace_invalid_positions(
    expected_output: List[Any],
    predicted_output: List[Any],
    value: Any
) -> (List[Any], List[Any]):
  """Return the expected and predicted outputs with the invalid positions replaced by the chosen value."""
  # Figure out the indices in expected output that are "BOS", "PAD" or None
  skip_indices = set([i for i, elem in enumerate(expected_output) if elem in [TRACR_BOS, TRACR_PAD, None]])

  # Replace such elements from expected and predicted output
  expected_output = [value if i in skip_indices else elem for i, elem in enumerate(expected_output)]
  predicted_output = [value if i in skip_indices else elem for i, elem in enumerate(predicted_output)]

  return expected_output, predicted_output

def replace_invalid_positions_in_expected_outputs(
  expected_outputs: List[List[Any]],
  predicted_outputs: Float[Tensor, "batch seq_len logits"],
  value: Any
) -> (List[List[Any]], Bool[Tensor, "batch seq_len"]):
  """Replaces the invalid positions in the expected outputs with the chosen value and returns the new expected outputs
  and a mask for the changed positions."""
  mask = t.full(predicted_outputs.shape[:-1], False, dtype=t.bool)

  new_expected_outputs = []
  for output_idx, expected_output in enumerate(expected_outputs):
    # Figure out the indices in expected output that are "BOS", "PAD" or None
    skip_indices = set([i for i, elem in enumerate(expected_output) if elem in [TRACR_BOS, TRACR_PAD, None]])

    # Replace such elements from expected and predicted output
    expected_output = [value if i in skip_indices else elem for i, elem in enumerate(expected_output)]
    new_expected_outputs.append(expected_output)

    # update mask
    mask[output_idx, t.tensor(list(skip_indices))] = True

  return new_expected_outputs, mask