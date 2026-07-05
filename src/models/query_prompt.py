from typing import List, Sequence

import torch
import torch.nn as nn


def compute_image_token_counts(image_grid_thw: torch.Tensor, merge_size: int) -> List[int]:
    if image_grid_thw is None:
        return []
    merge_length = merge_size**2
    return (image_grid_thw.prod(dim=-1) // merge_length).tolist()


def group_image_token_counts(token_counts: Sequence[int], num_images_per_sample: int) -> List[List[int]]:
    if num_images_per_sample <= 0:
        raise ValueError("num_images_per_sample must be > 0")
    if len(token_counts) % num_images_per_sample != 0:
        raise ValueError(
            f"Token counts length {len(token_counts)} must be divisible by num_images_per_sample {num_images_per_sample}"
        )
    return [
        list(token_counts[i : i + num_images_per_sample])
        for i in range(0, len(token_counts), num_images_per_sample)
    ]


def build_source_type_ids(
    token_counts_per_image: Sequence[int], max_source_images: int, device: torch.device
) -> torch.Tensor:
    parts = []
    for source_index, token_count in enumerate(token_counts_per_image):
        if token_count <= 0:
            continue
        parts.append(
            torch.full(
                (token_count,),
                min(source_index, max_source_images - 1),
                dtype=torch.long,
                device=device,
            )
        )
    if not parts:
        return torch.zeros(0, dtype=torch.long, device=device)
    return torch.cat(parts, dim=0)


class QueryFormerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn_query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn_kv_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_input = self.self_attn_norm(query_tokens)
        self_attn_output = self.self_attn(
            self_attn_input,
            self_attn_input,
            self_attn_input,
            need_weights=False,
        )[0]
        query_tokens = query_tokens + self_attn_output

        cross_query = self.cross_attn_query_norm(query_tokens)
        cross_kv = self.cross_attn_kv_norm(visual_tokens)
        cross_output = self.cross_attn(
            cross_query,
            cross_kv,
            cross_kv,
            key_padding_mask=~visual_mask.bool(),
            need_weights=False,
        )[0]
        query_tokens = query_tokens + cross_output

        query_tokens = query_tokens + self.mlp(self.mlp_norm(query_tokens))
        return query_tokens


class QueryPromptAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_query_tokens: int,
        num_heads: int,
        num_layers: int,
        num_input_layers: int,
        max_source_images: int,
    ):
        super().__init__()
        if num_input_layers <= 0:
            raise ValueError("num_input_layers must be > 0")

        self.max_source_images = max_source_images
        self.layer_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_input_layers)])
        self.layer_scale = nn.Parameter(torch.ones(num_input_layers))
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim * num_input_layers, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.source_type_embed = nn.Embedding(max_source_images, hidden_dim)
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [QueryFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        visual_layers: torch.Tensor,
        visual_mask: torch.Tensor,
        source_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        if visual_layers.ndim != 4:
            raise ValueError(
                f"visual_layers must be [batch, layers, seq, dim], got shape {tuple(visual_layers.shape)}"
            )
        if visual_mask.ndim != 2:
            raise ValueError(f"visual_mask must be [batch, seq], got shape {tuple(visual_mask.shape)}")
        if source_type_ids.ndim != 2:
            raise ValueError(f"source_type_ids must be [batch, seq], got shape {tuple(source_type_ids.shape)}")

        if visual_layers.shape[1] != len(self.layer_norms):
            raise ValueError(
                f"Expected {len(self.layer_norms)} visual layers, but got {visual_layers.shape[1]}"
            )

        normalized_layers = []
        for layer_index, layer_norm in enumerate(self.layer_norms):
            layer_tokens = layer_norm(visual_layers[:, layer_index])
            layer_tokens = layer_tokens * self.layer_scale[layer_index]
            normalized_layers.append(layer_tokens)

        stacked_visual = torch.cat(normalized_layers, dim=-1)
        visual_tokens = self.input_proj(stacked_visual)
        source_type_ids = source_type_ids.clamp_(0, self.max_source_images - 1)
        visual_tokens = visual_tokens + self.source_type_embed(source_type_ids)

        batch_size = visual_layers.shape[0]
        query_tokens = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        for block in self.blocks:
            query_tokens = block(query_tokens, visual_tokens, visual_mask)

        return self.output_proj(self.output_norm(query_tokens))
