from argparse import Namespace
from typing import Union

from transformer_lens.hook_points import HookedRootModule


def parse_dimension(dim_value: Union[int, None],
                    compression_ratio_value: Union[float, None],
                    max_size: int,
                    param_name: str) -> int:
  # Both can not be set at the same time
  assert dim_value is None or compression_ratio_value is None, \
    f"Both {param_name} and {param_name}_compression_ratio can not be set at the same time."

  if dim_value is None and compression_ratio_value is None:
    print(f"Warning: {param_name} and {param_name}_compression_ratio are not set. "
          f"Using the default value for this case: {max_size}.")
    return max_size

  if dim_value is not None:
    size = dim_value
  else:
    size = int(max_size * compression_ratio_value)

  assert 0 < size <= max_size, \
    f"Invalid {param_name} size: {size}. Size must be between 0 and {max_size}."

  return size


def parse_d_model(args: Namespace, tl_model: HookedRootModule):
  return parse_dimension(args.d_model, args.d_model_compression_ratio, tl_model.cfg.d_model, 'd_model')


def parse_d_head(args: Namespace, tl_model: HookedRootModule):
  return parse_dimension(args.d_head, args.d_head_compression_ratio, tl_model.cfg.d_head, 'd_head')
