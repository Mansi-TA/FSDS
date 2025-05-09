def test_package_imports():
    try:
        import FSDS_
        import FSDS_.feature
        import FSDS_.ingest
        import FSDS_.main
        import FSDS_.train
    except ImportError as e:
        assert False, f"Import failed{e}"
