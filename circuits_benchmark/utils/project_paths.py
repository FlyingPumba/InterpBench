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

  # Get the absolute path of the current file
  current_file_path = os.path.abspath(__file__)

  # Get the directory name of the current file
  current_dir = os.path.dirname(current_file_path)

  # Traverse upwards until you reach the root directory (assumed to be two levels up)
  PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, '..', '..'))

  return PROJECT_ROOT

def get_default_output_dir() -> str:
  """
  Get the default output directory for the project.
  :return: the default output directory for the project.
  """
  return str(os.path.join(detect_project_root(), "results"))
