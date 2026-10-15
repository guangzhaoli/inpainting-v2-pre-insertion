import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.balanced_sampler import CategoryBalancedSampler
from src.data.config import Config, ImageProcessorInitArgs
from src.data.preprocess import ImageProcessor, aspect_preserving_size
from src.loss.local_detail import foreground_detail_crops, interaction_masks
from src.loss.region_perceptual_loss import RegionPerceptualLoss
from src.utils.hybrid_masking import add_interaction_context
from src.utils.seed import prepare_trainable_precision, seed_training_rank


@pytest.mark.parametrize("size", [(942, 1255), (512, 768), (1920, 1080), (600, 1600), (800, 800)])
def test_pixel_budget_keeps_subject_proportions(size):
    w, h = size
    nw, nh = aspect_preserving_size(w, h, 1024**2)
    assert nw % 16 == nh % 16 == 0
    assert abs((nw / nh) / (w / h) - 1) < 0.025
    assert abs(nw * nh / 1024**2 - 1) < 0.035
    processor = ImageProcessor(ImageProcessorInitArgs(process_type="aspect_pixels", target_pixels=1024**2))
    assert processor.target_pixels == 1024**2


def test_image_and_mask_geometries_agree():
    processor = ImageProcessor(ImageProcessorInitArgs(process_type="aspect_pixels", target_pixels=65536))
    image = Image.fromarray(np.zeros((192, 100, 3), np.uint8))
    mask = Image.fromarray(np.full((192, 100), 255, np.uint8))
    result = processor.preprocess({"image": image, "mask": mask, "controls": [image], "control_masks": [mask]})
    assert result['image'].shape[-2:] == result['mask'].shape[-2:]
    assert result['controls'][0].shape[-2:] == result['control_masks'][0].shape[-2:]


def test_fp32_trainables_retain_small_adam_updates_and_frozen_backbone():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)).bfloat16()
    model[0].requires_grad_(False)
    prepare_trainable_precision(model)
    assert model[0].weight.dtype == torch.bfloat16
    assert model[1].weight.dtype == torch.float32
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-7)
    old = model[1].weight.detach().clone()
    model[1].weight.grad = torch.ones_like(model[1].weight)
    opt.step()
    assert torch.all(model[1].weight != old)
    assert opt.state[model[1].weight]['exp_avg'].dtype == torch.float32
    assert opt.state[model[1].weight]['exp_avg_sq'].dtype == torch.float32


def test_rank_randomness_and_checkpoint_replay():
    seed_training_rank(42, 0)
    first = torch.rand(24)
    seed_training_rank(42, 1)
    second = torch.rand(24)
    assert not torch.equal(first, second)
    state = torch.get_rng_state()
    expected = torch.rand(24)
    seed_training_rank(42, 2)
    torch.set_rng_state(state)
    assert torch.equal(expected, torch.rand(24))


def test_old_bf16_optimizer_state_restores_as_fp32():
    old_param = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    old_opt = torch.optim.AdamW([old_param])
    old_param.grad = torch.ones_like(old_param)
    old_opt.step()
    new_param = torch.nn.Parameter(old_param.detach().float())
    new_opt = torch.optim.AdamW([new_param])
    new_opt.load_state_dict(old_opt.state_dict())
    assert new_opt.state[new_param]['exp_avg'].dtype == torch.float32
    assert new_opt.state[new_param]['exp_avg_sq'].dtype == torch.float32


def test_balancing_covers_all_samples_and_is_epoch_deterministic():
    dataset = SimpleNamespace(samples=[{'category': 'a'}] * 9 + [{'category': 'b'}] * 2)
    sampler = CategoryBalancedSampler(dataset)
    values = list(sampler)
    assert set(values) == set(range(11))
    assert sum(i < 9 for i in values) == sum(i >= 9 for i in values) == 9
    assert values == list(sampler)
    sampler.set_epoch(1)
    assert values != list(sampler)


def test_interaction_hole_covers_gt_without_changing_supervision():
    gt = torch.zeros(1, 40, 40)
    gt[:, 16:24, 16:24] = 1
    before = gt.clone()
    hole = add_interaction_context(gt, gt, 4)
    assert torch.equal(gt, before)
    assert torch.all(hole >= gt) and hole.sum() > gt.sum()
    ring, far = interaction_masks(gt.unsqueeze(1), 4)
    assert ring.sum() > 0
    assert not torch.any(far * gt.unsqueeze(1))
    assert not torch.any(far * ring)


class PixelDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()), requires_grad=False)

    def forward(self, pred, target):
        return (pred - target).square().mean((1, 2, 3), keepdim=True) * self.scale


def test_detail_and_boundary_losses_backpropagate_in_their_regions():
    target = torch.zeros(1, 3, 64, 64)
    pred = torch.ones_like(target, requires_grad=True)
    mask = torch.zeros(1, 64, 64)
    mask[:, 24:40, 24:40] = 1
    loss = RegionPerceptualLoss(fg_metric='none', lpips_input_size=32)
    loss.lpips = PixelDistance()
    out = loss(pred, target, mask, fg_weight=0, bg_weight=0, detail_weight=1,
               detail_patch_size=16, detail_num_patches=3, boundary_weight=1,
               interaction_radius_px=4, pred_range='0-1', target_range='0-1')
    assert out['detail_lpips'] > 0 and out['boundary_l1'] > 0
    out['total'].backward()
    assert pred.grad[:, :, 20:44, 20:44].abs().sum() > 0
    assert pred.grad[:, :, :8, :8].abs().sum() == 0
    assert torch.isfinite(pred.grad).all()


def test_empty_foreground_does_not_create_detail_patches():
    x = torch.zeros(1, 3, 32, 32)
    a, b, owners = foreground_detail_crops(x, x, x[:, :1], 16, 3, 16)
    assert a is b is None and owners == []


def test_face_regions_exclude_unedited_faces_and_allow_missing_faces(monkeypatch):
    from src.data import face_regions

    detector = SimpleNamespace(detectMultiScale=lambda *a, **kw: [(4, 4, 20, 20), (36, 4, 24, 24)])
    monkeypatch.setattr(face_regions, '_detector', lambda: detector)
    image = torch.zeros(3, 64, 64)
    mask = torch.zeros(64, 64)
    mask[4:24, 4:24] = 1
    face = face_regions.face_region_mask(image, mask)
    assert face.sum() == 400
    assert face[:, 36:].sum() == 0
    assert face_regions.face_region_mask(image, mask * 0).sum() == 0


def test_missing_faces_skip_identity_encoder():
    criterion = RegionPerceptualLoss(fg_metric='none', enable_ref_dino=True)
    image = torch.rand(1, 3, 32, 32, requires_grad=True)
    empty = torch.zeros(1, 32, 32)
    loss = criterion._reference_dino_identity(image, empty, image, empty,
        pred_range='0-1', ref_range='0-1', sample_valid=None, sample_weights=None)
    assert loss == 0 and criterion.dino is None
    loss.backward()
    assert image.grad is not None and image.grad.abs().sum() == 0


def test_integrated_configs_enable_all_paths_with_local_assets():
    import yaml
    root = Path(__file__).resolve().parents[1]
    files = list((root / 'configs/teacher_v2').glob('*.yaml'))
    assert len(files) == 5
    for p in files:
        cfg = Config.model_validate(yaml.safe_load(p.read_text()))
        assert cfg.train.num_epochs == 5
        assert cfg.train.trainable_precision == 'float32' and cfg.train.rank_specific_rng
        assert cfg.model.zero_cond_t and cfg.model.attention_backend == 'flash_attention_2'
        assert cfg.data.init_args.processor.init_args.process_type == 'aspect_pixels'
        assert cfg.loss.ref_dino_weight > 0 and cfg.loss.detail_lpips_weight > 0
        assert cfg.loss.boundary_l1_weight > 0
        assert cfg.predict.latent_anchoring.mode.value == 'expanded_soft'
        assert cfg.logging.sampling.enable and cfg.logging.sampling.fixed_seed_across_steps
        assert cfg.data.init_args.hybrid_masking.interaction_dilation_px == cfg.loss.interaction_radius_px
        rows = json.loads(Path(cfg.logging.sampling.validation_data).read_text())
        assert len(rows) >= cfg.logging.sampling.num_samples
        assert all(Path(r[k]).is_file() for r in rows for k in ['ref_image', 'ref_mask', 'tar_image', 'tar_mask'])
