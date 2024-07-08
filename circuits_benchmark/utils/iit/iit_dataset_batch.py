from typing import Tuple

import torch as t

Inputs = t.Tensor
Targets = t.Tensor
BaseData = Tuple[Inputs, Targets]
AblationData = Tuple[Inputs, Targets]
IITDatasetBatch = Tuple[BaseData, AblationData]
