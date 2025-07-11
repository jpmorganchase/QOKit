import pathlib, subprocess, sys, pytest, importlib.util

# skip quickly if SciPy missing
if importlib.util.find_spec("scipy") is None:
    pytest.skip("SciPy not installed", allow_module_level=True)

ROOT = pathlib.Path(__file__).resolve().parents[1]   # project root
SCRIPT = ROOT / "scripts" / "benchmark_vs_bruteforce.py"

def test_bruteforce_cli_fast():
    subprocess.run(
        [sys.executable, str(SCRIPT), "--Ns", "16", "--ps", "1"],
        check=True, timeout=120
    )
