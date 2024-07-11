from .templates import SubCommand

subcommands = [
    SubCommand.ACDC.value, 
    SubCommand.EAP.value, 
    SubCommand.NODE_SP.value, 
    SubCommand.EDGE_SP.value, 
    SubCommand.INTEGRATED_GRADIENTS.value
]

thresholds = [
    0.0,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    0.025,
    0.05,
    0.1,
    0.2,
    0.5,
    0.8,
    1.0,
    10.0,
    20.0,
    50.0,
    100.0,
]