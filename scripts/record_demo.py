"""Run the offline CLI and save a reproducible process/memory receipt alongside its outputs."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import runpy
import subprocess
import sys
import time
from pathlib import Path

# Allow running directly from a checkout before installing editable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument('--receipt', required=True)
args, offline_args = parser.parse_known_args()
original_args = list(sys.argv)
sys.argv = ['personalplex-offline', *offline_args]
started = time.perf_counter()
runpy.run_module('personaplex_mlx.offline', run_name='__main__')
import mlx.core as mx
input_path = Path(offline_args[offline_args.index('--input-wav') + 1])
model_path = Path(offline_args[offline_args.index('--model') + 1])
receipt = {
    'command': ['python', *original_args],
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'platform': platform.platform(),
    'chip': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
    'python': platform.python_version(),
    'mlx': importlib.metadata.version('mlx'),
    'wall_seconds_including_loading': time.perf_counter() - started,
    'peak_mlx_memory_gib': mx.get_peak_memory() / 2**30,
    'peak_process_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
    'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
    'quantization': json.loads((model_path / 'config.json').read_text())['quantization'],
}
Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
Path(args.receipt).write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
