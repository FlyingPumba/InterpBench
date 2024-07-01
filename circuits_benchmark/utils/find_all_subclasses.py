import importlib
import pkgutil


def import_submodules(package, recursive: bool = True):
  """ Import all submodules of a module, recursively, including subpackages

  :param package: package (name or actual module)
  :type package: str | module
  :rtype: dict[str, types.ModuleType]
  """
  if isinstance(package, str):
    package = importlib.import_module(package)
  results = {}
  for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
    full_name = package.__name__ + '.' + name
    try:
      results[full_name] = importlib.import_module(full_name)
    except ModuleNotFoundError:
      continue
    if recursive and is_pkg:
      results.update(import_submodules(full_name))
  return results


def find_all_subclasses_in_package(base_class, package_name):
  """ Find all subclasses of a given class in a package.
  We first need to import all submodules of the package, then we can use the `__subclasses__` method of the base class.
  Otherwise, the subclasses might not be loaded yet, and we will miss them.
  """
  import_submodules(package_name)
  return base_class.__subclasses__()


def find_all_transitive_subclasses_in_package(base_class, package_name):
  """ Find all transitive subclasses of a given class in a package.
  """
  subclasses = set(find_all_subclasses_in_package(base_class, package_name))

  last_subclasses = None
  new_subclasses = subclasses.copy()
  while new_subclasses != last_subclasses:
    last_subclasses = new_subclasses.copy()
    for cls in last_subclasses:
      new_subclasses.update(cls.__subclasses__())

  return new_subclasses
