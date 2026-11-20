"""databucket1 / llama / reader: run against local inputs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from run_pipeline import main

if __name__ == "__main__":
    sys.exit(main(stage="reader", bucket="databucket1", model="llama"))
