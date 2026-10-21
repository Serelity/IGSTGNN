"""Record the actual server environment without changing installed packages."""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid


def command(arguments):
    try:
        completed = subprocess.run(arguments, capture_output=True, text=True,
                                   timeout=20, check=False)
        return dict(returncode=completed.returncode, stdout=completed.stdout.strip(),
                    stderr=completed.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(returncode=None, stdout='', stderr=str(error))


def collect(device='cuda:0'):
    import numpy as np
    import scipy
    import torch
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(dict(index=index, name=properties.name,
                                total_memory_bytes=properties.total_memory,
                                compute_capability=[properties.major, properties.minor]))
    package_versions = {}
    for package in ('numpy', 'scipy', 'torch', 'tqdm'):
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            package_versions[package] = None
    return dict(schema='pase_paper_server_environment_v1',
                recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                conda_environment=os.environ.get('CONDA_DEFAULT_ENV'),
                executable=sys.executable, python=platform.python_version(),
                platform=platform.platform(), packages=package_versions,
                torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                torch_cuda_runtime=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),
                requested_device=device, cuda_available=torch.cuda.is_available(),
                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu_devices=devices,
                nvidia_smi=command(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                    '--format=csv,noheader']),
                nvcc=command(['nvcc', '--version']),
                repository_requirements=dict(python='3.10 or 3.11', torch='2.3.1',
                                             cuda='not pinned'),
                version_scope='actual installed versions; driver, toolkit and torch CUDA runtime are distinct')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    environment = collect(args.device)
    if args.device.startswith('cuda') and not environment['cuda_available']:
        raise RuntimeError('CUDA is unavailable in the selected interpreter; activate igstgnn on a GPU node')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = 'environment_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.json'
    destination = args.output_dir/name
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(environment, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    gpu = '; '.join(item['name'] for item in environment['gpu_devices']) or 'CPU'
    print('Actual environment: Python '+environment['python']+', PyTorch '+environment['torch']
          +', torch CUDA '+str(environment['torch_cuda_runtime'])+', '+gpu, flush=True)
    print('Environment record: '+str(destination.resolve()), flush=True)


if __name__ == '__main__':
    main()
