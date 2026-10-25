def register():
    """Called in both the parent and vLLM worker by the general plugin loader."""
    from . import quantization  # noqa: F401
    from . import offline  # noqa: F401; opt-in offline weight/selection experiments
