import os


def detect_project_root() -> str:
  """
  Detects the root of the project by looking for a known file in the project.
  :return: the path to the root of the project.
  """
  current_path = os.path.abspath(os.path.curdir)
  while not os.path.exists(os.path.join(current_path, "pyproject.toml")):
    current_path = os.path.abspath(os.path.join(current_path, os.pardir))
  return current_path
