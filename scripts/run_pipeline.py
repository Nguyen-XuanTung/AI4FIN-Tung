"""Run consecutive research phases, stopping immediately on failure."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-phase', type=int, choices=range(1, 10), default=9)
    parser.add_argument('--end-phase', type=int, choices=range(1, 10), default=9)
    args = parser.parse_args()
    if args.start_phase > args.end_phase:
        parser.error('--start-phase must not exceed --end-phase')
    for phase in range(args.start_phase, args.end_phase + 1):
        subprocess.run([sys.executable, str(ROOT / f'scripts/run_phase{phase}.py')], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
