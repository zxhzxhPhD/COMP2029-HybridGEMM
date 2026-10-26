def register():
    """Called in both the parent and vLLM worker by the general plugin loader."""
    from . import quantization  # noqa: F401
    from . import offline  # noqa: F401; opt-in offline weight/selection experiments
    import os
    if os.environ.get('HYGEMM_RECORD_LOAD_TIMES','0')=='1':
        from .load_timing import install as install_load_timing
        install_load_timing()
    if os.environ.get('HYGEMM_FUSE_SILU','0')=='1':
        from .producer import install
        install()
        if os.environ.get('HYGEMM_FUSE_LLAMA','0')=='1':
            from .mistral_producer import install as install_mistral
            install_mistral()
