"""AnyFlow-style interpolated endpoint-time conditioning for Qwen."""

import copy
from torch import nn

FLOW_MAP_FILE = "flow_map_time_embedding.safetensors"


class FlowMapTimeEmbedding(nn.Module):
    def __init__(self, original_embedding, source_weight=0.25):
        super().__init__()
        if not 0 < source_weight < 1:
            raise ValueError("source_weight must be between zero and one")
        self.endpoint = copy.deepcopy(original_embedding)
        self.source_weight = float(source_weight)

    def forward(self, source_embedding, end_time, hidden_states, guidance=None):
        target = (
            self.endpoint(end_time, hidden_states)
            if guidance is None
            else self.endpoint(end_time, guidance, hidden_states)
        )
        return self.source_weight * source_embedding + (1 - self.source_weight) * target
