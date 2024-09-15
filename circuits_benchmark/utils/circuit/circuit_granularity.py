import typing
from typing import Literal

CircuitGranularity = Literal["component", "matrix", "acdc_hooks", "sp_hooks"]
circuit_granularity_options = list(typing.get_args(CircuitGranularity))
