from types import SimpleNamespace

import torch

from src.trainer.qwen_image_edit_trainer import QwenImageEditTrainer


class CacheVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.post_quant_conv = torch.nn.Conv3d(2, 2, 1)
        self.decoder = torch.nn.Conv3d(2, 3, 3, padding=1)
        self.requires_grad_(False)
        self.config = SimpleNamespace(z_dim=2, latents_mean=[0., 0.], latents_std=[1., 1.])
        self.cache = None
        self.calls = 0

    def decode(self, z, return_dict=False):
        self.cache = None
        self.calls += 1
        x = self.post_quant_conv(z)
        self.cache = x.detach()
        x = self.decoder(x).tanh()
        self.cache = None
        return (x,)


def test_checkpointed_decode_matches_outputs_and_gradients_and_clears_cache():
    trainer = object.__new__(QwenImageEditTrainer)
    trainer.vae = CacheVAE()
    trainer.vae_scale_factor = 1
    trainer._unpack_latents = lambda z, *args: z
    trainer.config = SimpleNamespace(train=SimpleNamespace(checkpoint_vae_decode=False))
    source = torch.rand(1, 2, 1, 8, 8)
    plain = source.clone().requires_grad_(True)
    expected = trainer.decode_vae_latent_for_loss(plain, 8, 8)
    expected.square().mean().backward()
    trainer.config.train.checkpoint_vae_decode = True
    checked = source.clone().requires_grad_(True)
    actual = trainer.decode_vae_latent_for_loss(checked, 8, 8)
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(checked.grad, plain.grad)
    assert trainer.vae.calls == 3 and trainer.vae.cache is None
    assert all(p.grad is None for p in trainer.vae.parameters())
