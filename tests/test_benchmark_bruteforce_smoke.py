import subprocess, sys, pathlib
import pytest, importlib.util
if importlib.util.find_spec("scipy") is None:
    pytest.skip("SciPy not installed", allow_module_level=True)

def test_bruteforce_cli_fast():
    subprocess.run(
        [sys.executable, "scripts/benchmark_vs_bruteforce.py",
         "--Ns", "16", "--ps", "1"], check=True, timeout=120
    )
    assert pathlib.Path("results/improvement_vs_bruteforce.csv").exists()
