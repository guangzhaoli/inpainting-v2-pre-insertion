import unittest
from unittest.mock import patch

import torch

from src.models.mask_geometry import MaskGeometryTokenizer, build_mask_geometry
from src.models.transformer_qwenimage import QwenImageTransformer2DModel


class MaskGeometryInjectionTest(unittest.TestCase):
    def _make_model(self):
        model = QwenImageTransformer2DModel(
            in_channels=8,
            out_channels=2,
            num_layers=0,
            attention_head_dim=8,
            num_attention_heads=1,
            joint_attention_dim=8,
            axes_dims_rope=(2, 2, 4),
        )
        model.mask_geometry_tokenizer = MaskGeometryTokenizer(
            hidden_channels=4,
            out_channels=8,
        )
        return model

    def test_zero_initialized_tokenizer_is_an_exact_model_noop(self):
        model = self._make_model()
        hidden_states = torch.randn(1, 6, 8)
        geometry = torch.randn(1, 3, 16, 16)
        img_shapes = [[(1, 2, 2), (1, 1, 2)]]

        actual = model._inject_mask_geometry(hidden_states, geometry, img_shapes)

        torch.testing.assert_close(actual, hidden_states, rtol=0, atol=0)

    def test_only_target_prefix_receives_geometry_residual(self):
        model = self._make_model()
        torch.nn.init.zeros_(model.mask_geometry_tokenizer.conv_out.weight)
        torch.nn.init.ones_(model.mask_geometry_tokenizer.conv_out.bias)
        hidden_states = torch.zeros(1, 6, 8)
        geometry = torch.randn(1, 3, 16, 16)
        img_shapes = [[(1, 2, 2), (1, 1, 2)]]

        actual = model._inject_mask_geometry(hidden_states, geometry, img_shapes)

        torch.testing.assert_close(actual[:, :4], torch.ones(1, 4, 8))
        torch.testing.assert_close(actual[:, 4:], torch.zeros(1, 2, 8))

    def test_geometry_without_attached_tokenizer_fails_clearly(self):
        model = self._make_model()
        del model.mask_geometry_tokenizer

        with self.assertRaisesRegex(ValueError, "no mask_geometry_tokenizer"):
            model._inject_mask_geometry(
                torch.zeros(1, 4, 8),
                torch.zeros(1, 3, 8, 8),
                [[(1, 2, 2)]],
            )

    def test_precomputed_tokens_reuse_tokenizer_output_without_retokenizing(self):
        model = self._make_model()
        torch.nn.init.normal_(model.mask_geometry_tokenizer.conv_out.weight)
        hidden_states = torch.randn(1, 6, 8)
        geometry = torch.randn(1, 3, 16, 16)
        img_shapes = [[(1, 2, 2), (1, 1, 2)]]

        expected = model._inject_mask_geometry(hidden_states, geometry, img_shapes)
        cached_tokens = model.prepare_mask_geometry_tokens(geometry, img_shapes)
        with patch.object(
            model.mask_geometry_tokenizer,
            "forward",
            side_effect=AssertionError("cached injection must not invoke tokenizer"),
        ):
            actual = model._inject_mask_geometry(
                hidden_states,
                None,
                img_shapes,
                mask_geometry_tokens=cached_tokens,
            )

        torch.testing.assert_close(actual, expected)

    def test_raw_and_precomputed_geometry_are_mutually_exclusive(self):
        model = self._make_model()
        with self.assertRaisesRegex(ValueError, "either mask_geometry or mask_geometry_tokens"):
            model._inject_mask_geometry(
                torch.zeros(1, 4, 8),
                torch.zeros(1, 3, 8, 8),
                [[(1, 2, 2)]],
                mask_geometry_tokens=torch.zeros(1, 4, 8),
            )

    def test_geometry_requires_target_shape_for_every_batch_item(self):
        model = self._make_model()
        with self.assertRaisesRegex(ValueError, "one img_shapes entry per batch item"):
            model._inject_mask_geometry(
                torch.zeros(2, 4, 8),
                torch.zeros(2, 3, 8, 8),
                [[(1, 2, 2)]],
            )

    def test_real_transformer_forward_backpropagates_into_zero_projection(self):
        model = QwenImageTransformer2DModel(
            in_channels=8,
            out_channels=2,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=1,
            joint_attention_dim=8,
            axes_dims_rope=(2, 2, 4),
        )
        model.mask_geometry_tokenizer = MaskGeometryTokenizer(
            hidden_channels=4,
            out_channels=8,
        )
        geometry = build_mask_geometry(
            torch.nn.functional.pad(torch.ones(1, 2, 2), (3, 3, 3, 3)),
            sigma_px=2.0,
        )

        output = model(
            hidden_states=torch.randn(1, 6, 8),
            encoder_hidden_states=torch.randn(1, 3, 8),
            encoder_hidden_states_mask=torch.ones(1, 3, dtype=torch.long),
            timestep=torch.tensor([0.5]),
            img_shapes=[[(1, 2, 2), (1, 1, 2)]],
            txt_seq_lens=[3],
            mask_geometry=geometry,
            return_dict=False,
        )[0]
        output[:, :4].sum().backward()

        self.assertEqual(tuple(output.shape), (1, 6, 8))
        gradient = model.mask_geometry_tokenizer.conv_out.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
