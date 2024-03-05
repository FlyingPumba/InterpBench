import typing
from typing import Literal

CompressionTrainLossLevel = Literal["layer", "component"]
compression_train_loss_level_options = list(typing.get_args(CompressionTrainLossLevel))