import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
import math
from typing import Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x
    
class flona_ViNT(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.floorplan_encoding_size = obs_encoding_size
        self.context_size = context_size

        # self.obs_floorplan_enc = nn.Linear(4, obs_encoding_size,device="cuda")
        # self.obs_floorplan_expand = nn.Linear(6, obs_encoding_size,device="cuda")
        self.obs_goal_pos_ori_enc = nn.Linear(obs_encoding_size + 6, obs_encoding_size) 
        # Initialize the observation encoder

        self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) 
        self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
        self.num_obs_features = self.obs_encoder._fc.in_features
        self.obs_encoder_type = "efficientnet"
       
        # Initialize the floorplan encoder
        self.floorplan_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) 
        self.floorplan_encoder = replace_bn_with_gn(self.floorplan_encoder)
        self.num_floorplan_features = self.floorplan_encoder._fc.in_features

        # Initialize compression layers if necessary
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        if self.num_floorplan_features != self.floorplan_encoding_size:
            self.compress_floorplan_enc = nn.Linear(self.num_floorplan_features, self.floorplan_encoding_size)
        else:
            self.compress_floorplan_enc = nn.Identity()

        # Initialize positional encoding and self-attention layers
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=self.context_size + 2)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size, 
            nhead=mha_num_attention_heads, 
            dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

    def forward(self, 
                obs_img: torch.tensor, 
                floorplan_img: torch.tensor, 
                obs_pos: torch.tensor, 
                goal_pos: torch.tensor, 
                obs_ori: torch.tensor, 
                # input_goal_mask: torch.tensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = obs_img.device
        obs_goal_pos = torch.cat([obs_pos, goal_pos], dim=1)
        obs_goal_pos_ori = torch.cat([obs_goal_pos, obs_ori], dim=1).to(device) 
        
        # Get the current obs and floorplan encoding
        obsfloorplan_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], floorplan_img], dim=1) # concatenate the last one in context (equal to current image) and floorplan image
        obsfloorplan_encoding = self.floorplan_encoder.extract_features(obsfloorplan_img) # get encoding of this img 
        obsfloorplan_encoding = self.floorplan_encoder._avg_pooling(obsfloorplan_encoding) # avg pooling 
        
        if self.floorplan_encoder._global_params.include_top:
            obsfloorplan_encoding = obsfloorplan_encoding.flatten(start_dim=1)
            obsfloorplan_encoding = self.floorplan_encoder._dropout(obsfloorplan_encoding)
        obsfloorplan_encoding = self.compress_floorplan_enc(obsfloorplan_encoding)

        if len(obsfloorplan_encoding.shape) == 2:
            obsfloorplan_encoding = obsfloorplan_encoding.unsqueeze(1)
        assert obsfloorplan_encoding.shape[2] == self.floorplan_encoding_size
        floorplan_encoding = obsfloorplan_encoding    # b, 1, obs_encoding_size
        
        # Get the observations encoding
        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.concat(obs_img, dim=0)

        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.unsqueeze(1)
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)      # b, context_size+1, obs_encoding_size
        obs_encoding = torch.cat((obs_encoding, floorplan_encoding), dim=1) # b, context_size+2, obs_encoding_size
        
        # Apply positional encoding 
        if self.positional_encoding:
            obs_encoding = self.positional_encoding(obs_encoding)
        obs_encoding_tokens = self.sa_encoder(obs_encoding, src_key_padding_mask=None)
        obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1)
        obs_encoding_tokens = torch.cat([obs_encoding_tokens, obs_goal_pos_ori], dim=1)
        obs_encoding_tokens = self.obs_goal_pos_ori_enc(obs_encoding_tokens) 
        return obs_encoding_tokens


# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module



    