def test_package_imports():
    try: 
        import FSDS_
        import FSDS_.feature
        import FSDS_.ingest
        import FSDS_.train
        import FSDS_.main
    except ImportError as e:
        assert False,f"Import failed{e}"