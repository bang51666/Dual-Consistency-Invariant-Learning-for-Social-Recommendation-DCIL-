import importlib.util
from pathlib import Path


TEST_FILE = Path(__file__).with_name("test_repository_contract.py")


def main():
    spec = importlib.util.spec_from_file_location("repository_contract", TEST_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    failed = 0
    total = 0
    for name in sorted(attr for attr in dir(module) if attr.startswith("test_")):
        total += 1
        try:
            getattr(module, name)()
            print(f"PASS {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{total - failed}/{total} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()