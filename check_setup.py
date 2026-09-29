"""
check_setup.py — run on every workstation after installation:   python check_setup.py
Checks: Python version, packages, data files, a 2-second PyTorch training, and that a local server can use port 8000.
Exit code 0 = ready.
"""
import importlib, socket, sys, time, hashlib
from pathlib import Path

OK, FAIL = "  [ OK ]", "  [FAIL]"
problems = 0
ROOT = Path(__file__).resolve().parent

def check(cond, msg, hint=""):
    global problems
    print((OK if cond else FAIL), msg, "" if cond else f"\n         → {hint}")
    problems += 0 if cond else 1

print("1. Python")
check(sys.version_info[:2] in [(3, 11), (3, 12)], f"Python {sys.version.split()[0]}", "Python 3.11 or 3.12 is required")

print("2. Packages")
for mod in ["torch", "numpy", "pandas", "matplotlib", "seaborn", "sklearn", "jupyterlab", "fastapi", "uvicorn", "pydantic", "requests"]:
    try:
        m = importlib.import_module(mod)
        check(True, f"{mod} {getattr(m, '__version__', '')}")
    except Exception as e:
        check(False, f"{mod} import", f"pip install -r requirements.txt  ({e})")

print("3. Data files")
EXPECTED = {"airport_traffic_2019.csv": 7_000_000, "airport_traffic_2023.csv": 7_000_000,
            "airport_traffic_2024.csv": 8_000_000, "airport_traffic_2025.csv": 8_000_000, "LIMC.csv": 8_500_000}
for name, min_size in EXPECTED.items():
    p = ROOT / "data" / name
    check(p.exists() and p.stat().st_size > min_size, f"data/{name}", "copy the data/ folder from the course package")

print("4. PyTorch training smoke test")
try:
    import torch
    torch.manual_seed(0)
    X = torch.randn(2000, 10); y = (X[:, 0] > 0).float()
    net = torch.nn.Sequential(torch.nn.Linear(10, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    t0 = time.time()
    for _ in range(200):
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X).squeeze(1), y); loss.backward(); opt.step()
    check(loss.item() < 0.2, f"tiny model trained in {time.time() - t0:.1f} s (loss {loss.item():.3f})", "PyTorch installation looks broken")
except Exception as e:
    check(False, "PyTorch smoke test", str(e))

print("5. Local server port")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # same behaviour as uvicorn
try:
    s.bind(("127.0.0.1", 8000)); check(True, "port 8000 on 127.0.0.1 is free and bindable")
except OSError as e:
    check(False, "port 8000 on 127.0.0.1", f"port busy or blocked ({e}); free it or allow local loopback servers")
finally:
    s.close()

print("\nRESULT:", "READY ✅" if problems == 0 else f"{problems} problem(s) ❌")
sys.exit(1 if problems else 0)
