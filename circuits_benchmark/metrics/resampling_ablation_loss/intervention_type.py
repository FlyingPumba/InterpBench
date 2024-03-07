from enum import Enum

from circuits_benchmark.training.compression.residual_stream_mapper.residual_stream_mapper import ResidualStreamMapper


class InterventionType(Enum):
  """Enum class for intervention type. We have 6 different types of interventions:
  - Regular corrupted data patching: we patch the clean run using the corresponding corrupted data on both models.
  - Corrupted data patching via compression: we take the corrupted data from the base model and use it to patch both
    models: the hypothesis model gets the corrupted data passed through a compressor, and the base model gets the
    corrupted data as it is.
  - Corrupted data patching via decompression: we take the corrupted data from the hypothesis model and use it to patch
    both models: the base model gets the corrupted data passed through a decompressor, and the hypothesis model gets the
    corrupted data as it is.
  - Clean data patching via compression: we take the clean data from the base model and use it to patch the hypothesis
    model after passing it through a compressor.
  - Clean data patching via decompression: we take the clean data from the hypothesis model and use it to patch the base
    model after passing it through a decompressor.
  - No intervention (the default): we let the clean data untouched on both models.

  Important note, some interventions are only possible if a residual stream mapper is provided. Otherwise, they should
    be skipped.
  """
  REGULAR_CORRUPTED = "REGULAR_CORRUPTED"
  CORRUPTED_COMPRESSION = "CORRUPTED_COMPRESSION"
  CORRUPTED_DECOMPRESSION = "CORRUPTED_DECOMPRESSION"
  CLEAN_COMPRESSION = "CLEAN_COMPRESSION"
  CLEAN_DECOMPRESSION = "CLEAN_DECOMPRESSION"
  NO_INTERVENTION = "NO_INTERVENTION"

  def __str__(self):
    return self.value

  @staticmethod
  def get_all():
    return [InterventionType.REGULAR_CORRUPTED, InterventionType.CORRUPTED_COMPRESSION,
            InterventionType.CORRUPTED_DECOMPRESSION, InterventionType.CLEAN_COMPRESSION,
            InterventionType.CLEAN_DECOMPRESSION, InterventionType.NO_INTERVENTION]

  @staticmethod
  def get_available_interventions(residual_stream_mapper: ResidualStreamMapper | None = None):
    """Returns the available interventions according to the provided residual_stream_mapper."""
    if residual_stream_mapper is None:
      return [InterventionType.REGULAR_CORRUPTED, InterventionType.NO_INTERVENTION]
    else:
      return InterventionType.get_all()