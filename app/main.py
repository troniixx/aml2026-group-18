"""
main.py — entry point for the Naruto Hand Seal Detector app.
 
Run from naruto_app/:
    python main.py
    python main.py --config path/to/other_config.yaml
"""
import argparse
import sys
from pathlib import Path
 
# Ensure the app folder is on sys.path regardless of where Python is invoked from
sys.path.insert(0, str(Path(__file__).parent))
 
from core.config import load as load_config
from core.detector import Detector
from ui.app import App
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Naruto Hand Seal Detector")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to config.yaml (default: naruto_app/config.yaml)",
    )
    args = parser.parse_args()
 
    cfg      = load_config(args.config)
    detector = Detector(cfg)
    app      = App(cfg, detector)
    app.run()
 
 
if __name__ == "__main__":
    main()
 