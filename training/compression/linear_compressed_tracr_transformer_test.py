import unittest

from benchmark.cases.case_3 import Case3
from training.compression.linear_compressed_tracr_transformer import LinearCompressedTracrTransformer


class LinearCompressedTracrTransformerTest(unittest.TestCase):
  def test_named_parameters_fold_compression_matrix(self):
    case = Case3()

    compressed_tracr_transformer = LinearCompressedTracrTransformer(
      case.get_tl_model(),
      int(9),
      "linear")

    list(compressed_tracr_transformer.named_parameters())