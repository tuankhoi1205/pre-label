"""Verify CUDA ops and checkpoint loading without running pre-label inference."""
import importlib.metadata
import json
from pathlib import Path
import sys

import torch
from mmcv.ops import Voxelization
from prelabel.config import load_config
from prelabel.detector import MMDet3DCenterPoint


def check():
    if not torch.cuda.is_available():
        raise RuntimeError('Docker runtime cannot access CUDA')
    points = torch.tensor([[2., 2., 1., 80., 0.], [2.01, 2.01, 1., 90., 0.]], device='cuda')
    voxelizer = Voxelization(voxel_size=[0.1, 0.1, 0.2],
        point_cloud_range=[-51.2, -51.2, -5., 51.2, 51.2, 3.],
        max_num_points=10, max_voxels=120000)
    voxels, coordinates, counts = voxelizer(points)
    torch.cuda.synchronize()
    if int(counts.sum()) != 2:
        raise RuntimeError('Compiled CUDA voxelization did not preserve test points')
    cfg = load_config('configs/mini.yaml')
    adapter = MMDet3DCenterPoint(cfg['detector'], 'runtime-validation-only')
    result = {'python': sys.version, 'gpu': torch.cuda.get_device_name(0),
        'cuda': torch.version.cuda, 'cuda_available': True,
        'mmcv_cuda_voxelization': True, 'centerpoint_checkpoint_loaded': True,
        'model': type(adapter.model).__name__, 'real_inference': False,
        'purpose': 'CUDA operator and model initialization only; no predictions/annotations',
        'packages': {p: importlib.metadata.version(p) for p in
            ['torch', 'torchvision', 'mmcv', 'mmengine', 'mmdet', 'mmdet3d', 'numpy',
             'scipy', 'nuscenes-devkit', 'matplotlib', 'Shapely']}}
    Path('artifacts/detector-runtime.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    check()
