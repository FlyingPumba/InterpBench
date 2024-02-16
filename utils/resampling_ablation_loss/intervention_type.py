from enum import Enum


class InterventionType(Enum):
  """Enum class for intervention type. We have 6 different types of interventions:
  - Regular corrupted data patching: we patch the clean run using the corresponding corrupted data on both models.
  - Corrupted data patching via encoding: we take the corrupted data from the base model and use it to patch both
    models: the hypothesis model gets the corrupted data passed through the encoder, and the base model gets the
    corrupted data as it is.
  - Corrupted data patching via decoding: we take the corrupted data from the hypothesis model and use it to patch
    both models: the base model gets the corrupted data passed through the decoder, and the hypothesis model gets the
    corrupted data as it is.
  - Clean data patching via encoding: we take the clean data from the base model and use it to patch the hypothesis
    model after passing it through the encoder.
  - Clean data patching via decoding: we take the clean data from the hypothesis model and use it to patch the base
    model after passing it through the decoder.
  - No intervention (the default): we let the clean data untouched on both models.

  Important note, interventions B-E are only possible if an autoencoder is provided. Otherwise, they should be skipped.
  """
  REGULAR_CORRUPTED = "REGULAR_CORRUPTED"
  CORRUPTED_ENCODING = "CORRUPTED_ENCODING"
  CORRUPTED_DECODING = "CORRUPTED_DECODING"
  CLEAN_ENCODING = "CLEAN_ENCODING"
  CLEAN_DECODING = "CLEAN_DECODING"
  NO_INTERVENTION = "NO_INTERVENTION"

  def __str__(self):
    return self.value

  @staticmethod
  def get_all():
    return [InterventionType.REGULAR_CORRUPTED, InterventionType.CORRUPTED_ENCODING, InterventionType.CORRUPTED_DECODING,
            InterventionType.CLEAN_ENCODING, InterventionType.CLEAN_DECODING, InterventionType.NO_INTERVENTION]

  @staticmethod
  def get_available_interventions(autoencoder):
    """Returns the available interventions according to the provided autoencoder."""
    if autoencoder is None:
      return [InterventionType.REGULAR_CORRUPTED, InterventionType.NO_INTERVENTION]
    else:
      return InterventionType.get_all()