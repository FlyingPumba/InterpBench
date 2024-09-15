class AttrDict(dict):
    """A dictionary that allows access to its values via attributes (e.g., using dot notation)."""

    def __getattr__(self, name):
        return self[name]
