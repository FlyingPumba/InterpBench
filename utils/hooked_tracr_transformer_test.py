import logging
import unittest

import jax

from benchmark.common_programs import make_reverse
from tracr.compiler import compiling
from tracr.rasp import rasp
from utils.hooked_tracr_transformer import HookedTracrTransformer

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update('jax_default_matmul_precision', 'float32')
logging.basicConfig(level=logging.ERROR)

class HookedTracrTransformerTest(unittest.TestCase):
  def test_no_exception(self):
    # Fetch RASP program
    program = make_reverse(rasp.tokens)

    # Compile it to a transformer model
    bos = "BOS"
    tracr_output = compiling.compile_rasp_to_model(
        program,
        vocab={1, 2, 3},
        max_seq_len=5,
        compiler_bos=bos,
    )
    tracr_model = tracr_output.model

    input = [bos, 1, 2, 3]
    print("Input:", input)

    tracr_output_decoded = tracr_model.apply(input).decoded
    print("Original Decoding:", tracr_output_decoded)

    tl_model = HookedTracrTransformer.from_tracr_model(tracr_model)
    tl_output_decoded = tl_model([input], return_type="decoded")[0]
    print("TransformerLens Replicated Decoding:", tl_output_decoded)

    self.assertEqual(tracr_output_decoded, tl_output_decoded)