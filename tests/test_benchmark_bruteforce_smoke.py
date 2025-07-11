import subprocess, sys, pathlib, importlib.util, pytest

# ── skip fast if SciPy (CLI dependency) is unavailable ─────────────────────
if importlib.util.find_spec("scipy") is None:
    pytest.skip("SciPy not installed", allow_module_level=True)

# ── build ABSOLUTE path to the benchmark script ────────────────────────────
ROOT   = pathlib.Path(__file__).resolve().parents[1]  # project root
SCRIPT = ROOT  / "scripts" / "benchmark_vs_bruteforce.py"

def test_bruteforce_cli_fast():
    assert SCRIPT.exists(), f"Script not found: {SCRIPT}"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--Ns", "16", "--ps", "1"],
        check=True, timeout=120
    )
