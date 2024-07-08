from iit.model_pairs.strict_iit_model_pair import StrictIITModelPair
from iit.utils import index


class TracrModelPair(StrictIITModelPair):
  @staticmethod
  def get_label_idxs():
    # Discard from all batches the first position, which is for the BOS token
    return index.Ix[:, 1:]
