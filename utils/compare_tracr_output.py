from typing import Any, List

import numpy as np
from jaxtyping import Float
from torch import Tensor


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
  This method only considers the positions that are not "BOS" or None in the expected output.
  """
  expected_output, predicted_output = remove_invalid_positions(expected_output, predicted_output)
  return compare_positions(expected_output, predicted_output, is_categorical, atol)


def compare_positions(expected_output, predicted_output, is_categorical, atol):
  if is_categorical:
    correct_positions = [elem1 == elem2 for elem1, elem2 in zip(predicted_output, expected_output)]
  else:
    # compare how close the outputs are numerically without taking into account the BOS token
    correct_positions = np.isclose(expected_output, predicted_output, atol=atol).tolist()
  return correct_positions


def remove_invalid_positions(
    expected_output: List[Any],
    predicted_output: List[Any] | Float[Tensor, ""]
) -> (List[Any], List[Any] | Float[Tensor, ""]):
  """Return the expected and predicted outputs without the invalid positions."""
  # Figure out the indices in expected output that are "BOS" or None
  skip_indices = set([i for i, elem in enumerate(expected_output) if elem in ["BOS", None]])

  # Remove such elements from expected and predicted output
  expected_output = [elem for i, elem in enumerate(expected_output) if i not in skip_indices]

  if isinstance(predicted_output, list):
    predicted_output = [elem for i, elem in enumerate(predicted_output) if i not in skip_indices]
  else:
    predicted_output = predicted_output[[i for i in range(len(predicted_output)) if i not in skip_indices]]

  return expected_output, predicted_output
