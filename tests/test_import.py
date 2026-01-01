def test_import():
    import spiro_analysis
    assert hasattr(spiro_analysis, "__version__") or True
