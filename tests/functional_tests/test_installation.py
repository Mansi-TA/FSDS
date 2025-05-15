def test_package_imports():
    try:
        import FSDS_
    except ImportError as e:
        assert False, f"Import failed{e}"
