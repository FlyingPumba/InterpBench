import os


PROJECT_ROOT: str | None = None


def detect_project_root() -> str:
  """
  Detects the root of the project by looking for a known file in the project.
  :return: the path to the root of the project.
  """
  global PROJECT_ROOT
  if PROJECT_ROOT is not None:
    # If the project root has already been detected, return it.
    return PROJECT_ROOT

  current_path = os.path.abspath(os.path.curdir)
  while not os.path.exists(os.path.join(current_path, "pyproject.toml")):
    current_path = os.path.abspath(os.path.join(current_path, os.pardir))

  PROJECT_ROOT = current_path

  return current_path


def get_default_output_dir() -> str:
  """
  Get the default output directory for the project.
  :return: the default output directory for the project.
  """
  return str(os.path.join(detect_project_root(), "results"))
