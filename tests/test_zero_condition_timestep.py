import copy
from types import MethodType

import pytest
import torch
import yaml

from src.data.config import Config, load_config_from_yaml
from src.models.mask_geometry import MaskGeometryTokenizer
from src.models.transformer_qwenimage import QwenImageTransformer2DModel
from src.utils.inference_metadata import validate_resume_mask_control_compatibility


def dense_modulation(self, x, parameters, split=None):
    # Independent per-token oracle: noisy target, clean reference/source.
    shift, scale, gate = (p[:, None] for p in parameters.chunk(3, dim=-1))
    if split is not None:
        mask = torch.arange(x.shape[1])[None, :, None] < split
        shift, scale, gate = (
            torch.where(mask, p.chunk(2, dim=0)[0], p.chunk(2, dim=0)[1])
            for p in (shift, scale, gate)
        )
    return x * (1 + scale) + shift, gate


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_native_condition_time_matches_dense_values_and_gradients(batch, checkpointed):
    torch.manual_seed(5)
    model = QwenImageTransformer2DModel(
        num_layers=2, attention_head_dim=16, num_attention_heads=2,
        in_channels=16, out_channels=4, joint_attention_dim=16,
        axes_dims_rope=(4, 6, 6), zero_cond_t=True,
    )
    model.mask_geometry_tokenizer = MaskGeometryTokenizer(4, 16)
    oracle = copy.deepcopy(model)
    for block in oracle.transformer_blocks:
        block._modulate = MethodType(dense_modulation, block)
    if checkpointed:
        model.enable_gradient_checkpointing()
        oracle.enable_gradient_checkpointing()
    source = torch.randn(batch, 8, 16)
    context = torch.randn(batch, 3, 16)
    times = torch.linspace(.2, .8, batch)
    observed = []
    model.time_text_embed.time_proj.register_forward_pre_hook(lambda _m, args: observed.append(args[0].clone()))
    kwargs = dict(encoder_hidden_states=context, encoder_hidden_states_mask=torch.ones(batch, 3),
                  timestep=times, img_shapes=[[(1, 2, 2), (1, 2, 2)]]*batch,
                  txt_seq_lens=[3]*batch, mask_geometry=torch.randn(batch, 3, 32, 32), return_dict=False)
    left, right = source.clone().requires_grad_(), source.clone().requires_grad_()
    actual, expected = model(left, **kwargs)[0], oracle(right, **kwargs)[0]
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(left.grad, right.grad)
    for (_, p), (_, q) in zip(model.named_parameters(), oracle.named_parameters()):
        if p.grad is not None or q.grad is not None:
            torch.testing.assert_close(p.grad, q.grad)
    torch.testing.assert_close(observed[0], torch.cat((times, torch.zeros_like(times))))


def test_old_checkpoint_time_contract_is_explicit_and_resume_cannot_switch(tmp_path):
    cfg = Config.model_validate({"predict":{"devices":{}}})
    assert cfg.model.zero_cond_t is False
    raw = cfg.model_dump(mode="json")
    raw["model"].pop("zero_cond_t")
    path = tmp_path / "train_config.yaml"
    path.write_text(yaml.safe_dump(raw))
    loaded = load_config_from_yaml(str(path))
    assert loaded.model.zero_cond_t is False
    cfg.model.zero_cond_t = True
    with pytest.raises(ValueError, match="zero_cond_t mismatch"):
        validate_resume_mask_control_compatibility(cfg, tmp_path)
