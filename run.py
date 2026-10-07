"""Launch API server and Gradio UI together."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.config import API_HOST, API_PORT, GRADIO_PORT


def main() -> None:
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "app.main"],
        cwd=str(ROOT),
    )
    print(f"API started (PID {api_proc.pid}) at http://{API_HOST}:{API_PORT}")
    time.sleep(3)

    ui_proc = subprocess.Popen(
        [sys.executable, "-m", "app.ui.gradio_app"],
        cwd=str(ROOT),
    )
    print(f"Gradio UI started (PID {ui_proc.pid}) at http://127.0.0.1:{GRADIO_PORT}")

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        api_proc.terminate()
        ui_proc.terminate()


if __name__ == "__main__":
    main()
