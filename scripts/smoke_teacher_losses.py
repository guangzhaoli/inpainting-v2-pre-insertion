"""Offline real-weight backward smoke test; this does not load/train the Qwen DiT."""

import json
from pathlib import Path

import torch

from src.data.config import load_config_from_yaml
from src.data.insert_anything_dataset import InsertAnythingDataset
from src.loss.region_perceptual_loss import RegionPerceptualLoss


def main():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    cfg = load_config_from_yaml(str(root / 'configs/teacher_v2/human.yaml'))
    loss = cfg.loss
    criterion = RegionPerceptualLoss(
        fg_metric=loss.fg_metric, lpips_net=loss.lpips_net, lpips_input_size=64,
        lpips_linear_path=loss.lpips_linear_path, lpips_torch_home=loss.lpips_torch_home,
        lpips_trunk_path=loss.lpips_trunk_path, enable_ref_dino=True,
        dino_backend=loss.dino_backend, dino_model_name=loss.dino_model_name,
    )
    # Only the smoke graph uses small synthetic images. Training configs retain
    # their 1 MP budget, full datasets, epochs and full-size loss settings.
    generator = torch.Generator().manual_seed(42)
    target = torch.rand(1, 3, 96, 96, generator=generator)
    pred = (target * .95 + .02).detach().requires_grad_(True)
    mask = torch.zeros(1, 96, 96)
    mask[:, 16:80, 16:80] = 1
    out = criterion(pred, target, mask, fg_weight=.5, bg_weight=.5, ref_dino_weight=.2,
                    ref_images=target, ref_mask=mask, pred_range='0-1', target_range='0-1',
                    detail_weight=.25, detail_num_patches=2, detail_patch_size=48,
                    boundary_weight=.25, interaction_radius_px=8,
                    sample_weights=torch.tensor([.64]))
    out['total'].backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    assert all(p.grad is None for p in criterion.parameters())
    assert all(torch.isfinite(v) for v in out.values())
    result = {'losses': {k: float(v.detach()) for k, v in out.items()},
              'input_gradient_l1': float(pred.grad.abs().sum()), 'frozen_encoders_have_no_grads': True,
              'smoke_input_size': [96, 96], 'training_model_loaded': False, 'samples': {}}
    for config_path in sorted((root / 'configs/teacher_v2').glob('*.yaml')):
        config = load_config_from_yaml(str(config_path))
        dataset = InsertAnythingDataset(config.data.init_args)
        sample = dataset[0]
        assert sample['tar_image'].shape[-2:] == sample['tar_mask'].shape[-2:]
        assert sample['ref_image'].shape[-2:] == sample['ref_mask'].shape[-2:]
        result['samples'][config_path.stem] = {
            'n_training_samples': len(dataset), 'target_shape': list(sample['tar_image'].shape),
            'reference_shape': list(sample['ref_image'].shape),
            'has_face_masks': 'target_face_mask' in sample,
        }
    output = root / 'output/teacher_v2_loss_smoke.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
