import unittest

import torch as t

from circuits_benchmark.commands.build_main_parser import build_main_parser
from circuits_benchmark.commands.train import train


class TrainTest(unittest.TestCase):
  def test_linear_compression(self):
    args, _ = build_main_parser().parse_known_args(["train",
                                                    "linear-compression",
                                                    "-i=2,3",
                                                    "--d-model=5",
                                                    "--epochs=1",
                                                    "--max-train-samples=10",
                                                    "--min-train-samples=10",
                                                    "--test-data-ratio=0.3",
                                                    "--batch-size=2",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)

  def test_non_linear_compression_works(self):
    args, _ = build_main_parser().parse_known_args(["train",
                                                    "non-linear-compression",
                                                    "-i=2,3",
                                                    "--d-model=8",
                                                    "--max-train-samples=10",
                                                    "--min-train-samples=10",
                                                    "--test-data-ratio=0.3",
                                                    "--epochs=1",
                                                    "--ae-epochs=2",
                                                    "--ae-max-train-samples=5",
                                                    "--resample-ablation-test-loss=True",
                                                    "--resample-ablation-max-interventions=1",
                                                    "--device=" + ("cuda" if t.cuda.is_available() else "cpu")])
    train.run(args)
