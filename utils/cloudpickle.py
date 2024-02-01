import os

from cloudpickle import cloudpickle


def load_from_pickle(path) -> object | None:
  if os.path.exists(path):
    with open(path, "rb") as f:
      return cloudpickle.load(f)
  else:
    return None


def dump_to_pickle(path, obj) -> None:
  with open(path, "wb") as f:
    cloudpickle.dump(obj, f)