from typing import Optional, Tuple
from dataclasses import dataclass
from transformers import EfficientNetForImageClassification
import math
from torch import nn
import torch
from transformers.utils.generic import ModelOutput
from transformers.models.efficientnet.modeling_efficientnet import EfficientNetEncoder


# Following two functions taken from modeling_efficientnet.py
def round_repeats(repeats, depth_coefficient):
    # Round number of block repeats based on depth multiplier.
    return int(math.ceil(depth_coefficient * repeats))


def round_filters(config, num_channels: int):
    r"""
    Round number of filters based on depth multiplier.
    """
    divisor = config.depth_divisor
    num_channels *= config.width_coefficient
    new_dim = max(divisor, int(num_channels + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_dim < 0.9 * num_channels:
        new_dim += divisor

    return int(new_dim)


@dataclass
class BaseModelOutputWithNoAttention(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class EfficientNetAdapterEncoding(EfficientNetEncoder):
    def __init__(self, config, model, adapter_base_block_idx,adapter_weight=0):
        encoder_instance = model.efficientnet.encoder
        self.config = model.config
        super().__init__(self.config)
        self.adapter_weight = adapter_weight
        self.blocks = encoder_instance.blocks
        self.top_conv = encoder_instance.top_conv
        self.top_bn = encoder_instance.top_bn
        self.top_activation = encoder_instance.top_activation
        self.adapter_idxs = [] # idx of block before adapter
        self.adapter_base_block_idx = (
            adapter_base_block_idx  # base block idx before adapter
        )
        self.gradient_checkpointing = False  # Unsure if this correctly sets gradient_checkpointing which seems to be defualt false for EncoderBlocks

        adapters = []

        block_class = self.blocks[0].__class__
        num_base_blocks = len(config.in_channels)
        block_dimensions = []
        block_idx = 0
        for i in range(num_base_blocks):
            block_out_dim = round_filters(config, config.out_channels[i])
            block_in_dim = round_filters(config, config.in_channels[i])
            for j in range(
                round_repeats(config.num_block_repeats[i], config.depth_coefficient)
            ):
                block_in_dim = block_out_dim if j > 0 else block_in_dim
                block_dimensions.append((block_in_dim, block_out_dim))
                block_idx += 1
            if i in self.adapter_base_block_idx:
                self.adapter_idxs.append(block_idx-1)

        for idx, base_idx in zip(self.adapter_idxs, self.adapter_base_block_idx):
            adapter_dimension = block_dimensions[idx]
            adapter = block_class(
                config=config,
                in_dim=adapter_dimension[0],
                out_dim=adapter_dimension[1],
                stride=config.strides[base_idx],
                kernel_size=config.kernel_sizes[base_idx],
                expand_ratio=1,  # No expansion
                drop_rate=config.drop_connect_rate * idx / len(self.blocks),
                id_skip=False,  # This is only true for initial blocks in base blocks
                adjust_padding=True,  # Always true for this model, since no block num in config.depthwise_padding
            )
            adapters.append(adapter)
        self.adapters = nn.ModuleList(adapters)
        assert self.adapters[0].__class__ == self.blocks[0].__class__
        assert len(self.adapters) == len(self.adapter_idxs)

    def forward(
        self, hidden_states, output_hidden_states=False, return_dict=True
    ) -> BaseModelOutputWithNoAttention:
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        adapters = iter(self.adapters)
        for idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if idx in self.adapter_idxs:
                adapter = next(adapters)
                adapter_hidden_states = adapter(hidden_states)
                if output_hidden_states:
                    all_hidden_states += (adapter_hidden_states,)
                if self.adapter_weight:
                    hidden_states = (self.adapter_weight * adapter_hidden_states + hidden_states)/2
                else:
                    hidden_states = adapter_hidden_states

        hidden_states = self.top_conv(hidden_states)
        hidden_states = self.top_bn(hidden_states)
        hidden_states = self.top_activation(hidden_states)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        assert hidden_states is not None
        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


def get_adapter_model(
    model_name: str,  # ex: "google/efficientnet-b3"
    adapter_locations: list[int],  # list of (base_block_idx,adapter_block_idx)
):
    adapter_locations.sort()
    adapter_base_block_idx = adapter_locations
    model = EfficientNetForImageClassification.from_pretrained(model_name)
    config = model.config
    adapter_encoding = EfficientNetAdapterEncoding(
        config, model, adapter_base_block_idx
    )
    model.efficientnet.encoder = adapter_encoding
    return model


def print_block_layout(config):
    """Valid block numbers shows options for adapter locations when using get_adapter_model"""
    num_base_blocks = len(config.in_channels)
    block_number = -1
    for i in range(num_base_blocks):
        in_dim = round_filters(config, config.in_channels[i])
        out_dim = round_filters(config, config.out_channels[i])

        base_block = []
        for j in range(
            round_repeats(config.num_block_repeats[i], config.depth_coefficient)
        ):
            in_dim = out_dim if j > 0 else in_dim
            base_block.append((in_dim, out_dim))
            block_number += 1
        print(f"Base block {i}: {base_block} valid block numbers: {(i,block_number)}")
    print(
        f"Avoid block numbers {config.depthwise_padding} since they have adjust_padding=Falsae"
    )
