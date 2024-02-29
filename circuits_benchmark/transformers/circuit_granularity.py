import typing
from typing import Literal

CircuitGranularity = Literal["component", "matrix"]
circuit_granularity_options = list(typing.get_args(CircuitGranularity))