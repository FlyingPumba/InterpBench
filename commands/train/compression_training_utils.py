from utils.hooked_tracr_transformer import HookedTracrTransformer


def parse_compression_size(args, tl_model: HookedTracrTransformer):
  compression_size = args.residual_stream_compression_size
  if compression_size == "auto":
    return compression_size

  # separate by commas and convert to integers
  compression_size = [int(size.strip()) for size in compression_size.split(",")]

  assert all(0 < size <= tl_model.cfg.d_model for size in compression_size), \
    f"Invalid residual stream compression size: {str(compression_size)}. " \
      f"All sizes in a comma separated list must be between 0 and {tl_model.cfg.d_model}."

  assert len(compression_size) > 0, "Must specify at least one residual stream compression size."

  return compression_size