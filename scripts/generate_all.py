import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
GEN_SCRIPTS = sorted(SCRIPTS_DIR.glob("gen_*.py"))

if __name__ == "__main__":
    for script in GEN_SCRIPTS:
        print(f"--- running {script.name} ---")
        subprocess.run([sys.executable, str(script)], check=True, cwd=SCRIPTS_DIR.parent)
    print("\nAll figures regenerated in figures/")
