import os

PROJECT_FOLDER = "circuits-benchmark"


def relativize_path(path) -> str:
  """Relativizes the path to the project root."""
  cwd = os.getcwd()
  parts = cwd.split("/")
  parts.reverse()

  for part in parts:
    if part == PROJECT_FOLDER:
      break
    else:
      path = os.path.join("..", path)
  return path