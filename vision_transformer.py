1# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attention:
            return x, attn
        else:
            return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x, return_all_patches=False):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if return_all_patches:
            return x
        else:
            return x[:, 0]

    def forward_all(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                x, attn = blk(x, return_attention=True)
                x = self.norm(x)
                return x, attn

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class HASHHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256, code_dim=32, class_num=200):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            layers.append(nn.BatchNorm1d(bottleneck_dim)) ## new
            layers.append(nn.GELU()) ## new
            self.mlp = nn.Sequential(*layers)
            
        self.hash = nn.Linear(bottleneck_dim, code_dim, bias=False)
        self.variance = nn.Linear(bottleneck_dim, code_dim, bias=False)
        self.bn_h = nn.BatchNorm1d(code_dim)
        self.bn_v = nn.BatchNorm1d(code_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        h = self.hash(x)
        v = self.variance(x)
        
        h = self.bn_h(h)
        v = self.bn_v(v)
        
        v = v / (nn.Tanh()(v * 1))
        h = nn.Tanh()(h * 1) 
        
        x = h * v

        return x, h, v
    
    
class BASEHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256, code_dim=32, class_num=200):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            layers.append(nn.BatchNorm1d(bottleneck_dim)) ## new
            layers.append(nn.GELU()) ## new
            self.mlp = nn.Sequential(*layers)
            self.bn = nn.BatchNorm1d(code_dim)
            
        self.head = nn.Linear(bottleneck_dim, code_dim, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = self.head(x)
        
        x = self.bn(x)
    
        return x, x, x


class VisionTransformerWithLinear(nn.Module):

    def __init__(self, base_vit, num_classes=200):

        super().__init__()

        self.base_vit = base_vit
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x, return_features=False):

        features = self.base_vit(x)
        features = torch.nn.functional.normalize(features, dim=-1)
        logits = self.fc(features)

        if return_features:
            return logits, features
        else:
            return logits

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.fc.weight.data.clone()
        w = torch.nn.functional.normalize(w, dim=1, p=2)
        self.fc.weight.copy_(w)




class PPNet_Normal(nn.Module):

    def __init__(self, features, img_size, prototype_shape,
                 num_classes,
                 use_global=False,
                 use_ppc_loss=False,
                 ppc_cov_thresh=2.,
                 ppc_mean_thresh=2,
                 global_coe=0.3,
                 global_proto_per_class=10,
                 init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(PPNet_Normal, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes

        self.use_global = use_global
        self.use_ppc_loss = use_ppc_loss
        self.ppc_cov_thresh = ppc_cov_thresh
        self.ppc_mean_thresh = ppc_mean_thresh
        self.global_coe = global_coe
        self.global_proto_per_class = global_proto_per_class
        self.epsilon = 1e-4
        


        self.num_prototypes_global = self.num_classes * self.global_proto_per_class
        # self.num_prototypes_global = self.num_classes
        self.prototype_shape_global = [self.num_prototypes_global] + self.prototype_shape[1:]

        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)
        self.prototype_class_identity_global = torch.zeros(self.num_prototypes_global,
                                                    self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        num_prototypes_per_class_global = self.num_prototypes_global // self.num_classes
        for j in range(self.num_prototypes_global):
            self.prototype_class_identity_global[j, j // num_prototypes_per_class_global] = 1


        # this has to be named features to allow the precise loading
        self.features = features


        first_add_on_layer_in_channels = [i for i in features.modules() if isinstance(i, nn.Linear)][-1].out_features

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)
        if self.use_global:
            self.prototype_vectors_global = nn.Parameter(torch.rand(self.prototype_shape_global),
                                              requires_grad=True)

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias
        self.last_layer_global = nn.Linear(self.num_prototypes_global, self.num_classes,
                                    bias=False) # do not use bias
        self.last_layer.weight.requires_grad = False
        self.last_layer_global.weight.requires_grad = False

        self.all_attn_mask = None
        self.teacher_model = None

        self.scale = self.prototype_shape[1] ** -0.5

        if init_weights:
            self._initialize_weights()

        # self.my_last = nn.Linear(self.prototype_shape[1], 100, bias=False)
        # nn.init.kaiming_normal_(self.my_last.weight, mode='fan_out', nonlinearity='relu')

        self.rankstat_head = nn.Linear(first_add_on_layer_in_channels, 48, bias=False)
        nn.init.kaiming_normal_(self.rankstat_head.weight, mode='fan_out', nonlinearity='relu')

        # self.my_last = nn.Linear(2000, 100, bias=False)
        # self.my_last_global = nn.Linear(2000, 100, bias=False)

        # trunc_normal_(self.my_last.weight, std=.02)
        # trunc_normal_(self.my_last_global.weight, std=.02)

        # 遍历 protopformer.features.blocks
        # 首先关闭 self.protopformer.features 中所有参数的梯度
        # for param in self.features.parameters():
        #     param.requires_grad = False
        
        # # self.add_on_layers.requires_grad = False

        # # 然后重新开启最后一个block的梯度
        # for param in self.features.blocks[-1].parameters():
        #     param.requires_grad = True


    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution_single(self, x, prototype_vectors):
        temp_ones = torch.ones(prototype_vectors.shape).cuda()

        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=temp_ones)

        p2 = prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances


    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def get_activations(self, tokens, prototype_vectors):
        batch_size, num_prototypes = tokens.shape[0], prototype_vectors.shape[0]
        distances = self._l2_convolution_single(tokens, prototype_vectors)
        activations = self.distance_2_similarity(distances)   # (B, 2000, 1, 1)
        total_proto_act = activations
        fea_size = activations.shape[-1]
        if fea_size > 1:
            activations = F.max_pool2d(activations, kernel_size=(fea_size, fea_size))   # (B, 2000, 1, 1)
        activations = activations.reshape(batch_size, num_prototypes)
        if self.use_global:
            return activations, (distances, total_proto_act)
        return activations



    def forward(self, x):

        # if not self.training:
        #     ##dino
        #     cls_tokens, img_tokens = self.features.forward_all(x)
        #     ##deit
        #     # all_tokens = self.features.forward_features_all(x)
        #     # cls_tokens, img_tokens = all_tokens[:, :1], all_tokens[:, 1:]
        #     # print("cls_tokens.shape", cls_tokens.shape)
        #     # print("img_tokens.shape", img_tokens.shape)

        #     B, dim = cls_tokens.shape[0], cls_tokens.shape[2] 
        #     fea_height, fea_width = int(img_tokens.shape[1] ** (1/2)), int(img_tokens.shape[1] ** (1/2))
        #     cls_tokens = cls_tokens.permute(0, 2, 1).reshape(B, dim, 1, 1) 
        #     img_tokens = img_tokens.permute(0, 2, 1).reshape(B, dim, fea_height, fea_width) 
        #     # print("cls_tokens.shape", cls_tokens.shape)
        #     # print("img_tokens.shape", img_tokens.shape)

        #     cls_tokens = self.add_on_layers(cls_tokens)
        #     img_tokens = self.add_on_layers(img_tokens)

        #     global_activations, _ = self.get_activations(cls_tokens, self.prototype_vectors_global)
        #     local_activations, _ = self.get_activations(img_tokens, self.prototype_vectors)

        #     logits_global = self.last_layer_global(global_activations)
        #     logits_local = self.last_layer(local_activations)
        #     logits = self.global_coe * logits_global + (1. - self.global_coe) * logits_local
        #     # activations = torch.cat([global_activations, local_activations], dim=1)
        #     # feature = self.my_last(activations)
        #     return logits, (_, _, logits_global, logits_local)

        # re-calculate distances
        ##dino
        cls_tokens, img_tokens = self.features.forward_all(x)
        ## deit
        # all_tokens = self.features.forward_features_all(x)
        # cls_tokens, img_tokens = all_tokens[:, :1], all_tokens[:, 1:]
        # print("cls_tokens.shape", cls_tokens.shape)
        # print("img_tokens.shape", img_tokens.shape)
        ##my_rankstat
        feat = self.rankstat_head(cls_tokens.squeeze(1))
        # feat = torch.nn.functional.normalize(feat, dim=-1)
        ##my_rankstat

        B, dim = cls_tokens.shape[0], cls_tokens.shape[2] 
        fea_height, fea_width = int(img_tokens.shape[1] ** (1/2)), int(img_tokens.shape[1] ** (1/2))
        cls_tokens = cls_tokens.permute(0, 2, 1).reshape(B, dim, 1, 1) 
        img_tokens = img_tokens.permute(0, 2, 1).reshape(B, dim, fea_height, fea_width) 
        # print("cls_tokens.shape", cls_tokens.shape)
        # print("img_tokens.shape", img_tokens.shape)

        # cls_tokens = self.add_on_layers(cls_tokens)
        img_tokens = self.add_on_layers(img_tokens)

        ##mylast
        # print("cls_tokens.shape", cls_tokens.shape)
        # logits_backbone = self.my_last(cls_tokens.squeeze())
        ##mylast

        # global_activations, _ = self.get_activations(cls_tokens, self.prototype_vectors_global)
        local_activations, _ = self.get_activations(img_tokens, self.prototype_vectors)

        # logits_global = self.last_layer_global(global_activations)
        logits_local = self.last_layer(local_activations)
        # logits = self.global_coe * logits_global + (1. - self.global_coe) * logits_local

        # print("global_activations.shape", global_activations.shape)
        # print("local_activations.shape", local_activations.shape)
        # activations = torch.cat([global_activations, local_activations], dim=1)
        # feature = self.my_last(activations)

        # return logits, global_activations, local_activations
        return logits_local, local_activations, feat

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        reserve_layer_nums = self.reserve_layer_nums
        (cls_tokens, img_tokens), (token_attn, cls_token_attn, _) = self.prototype_distances(x, reserve_layer_nums)
        global_activations, _ = self.get_activations(cls_tokens, self.prototype_vectors_global)
        local_activations, (distances, proto_acts) = self.get_activations(img_tokens, self.prototype_vectors)

        return cls_token_attn, proto_acts


    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

        if hasattr(self, 'last_layer_global'):
            positive_one_weights_locations = torch.t(self.prototype_class_identity_global)
            negative_one_weights_locations = 1 - positive_one_weights_locations

            self.last_layer_global.weight.data.copy_(
                correct_class_connection * positive_one_weights_locations
                + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


class MyHASHHead(nn.Module):
    def __init__(self, backbone, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256, code_dim=32, class_num=200):
        super().__init__()
        # nlayers = max(nlayers, 1)
        # if nlayers == 1:
        #     self.mlp = nn.Linear(in_dim, bottleneck_dim)
        # else:
        #     layers = [nn.Linear(in_dim, hidden_dim)]
        #     if use_bn:
        #         layers.append(nn.BatchNorm1d(hidden_dim))
        #     layers.append(nn.GELU())
        #     for _ in range(nlayers - 2):
        #         layers.append(nn.Linear(hidden_dim, hidden_dim))
        #         if use_bn:
        #             layers.append(nn.BatchNorm1d(hidden_dim))
        #         layers.append(nn.GELU())
        #     layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        #     layers.append(nn.BatchNorm1d(bottleneck_dim)) ## new
        #     layers.append(nn.GELU()) ## new
        #     self.mlp = nn.Sequential(*layers)
            
        self.hash = nn.Linear(in_dim, code_dim, bias=False)
        # self.variance = nn.Linear(in_dim, code_dim, bias=False)
        self.bn_h = nn.BatchNorm1d(code_dim)
        # self.bn_v = nn.BatchNorm1d(code_dim)
        
        self.apply(self._init_weights)
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # for name, m in self.backbone.features.named_parameters():
        #     if 'block' in name:
        #         block_num = int(name.split('.')[1])
        #         if block_num >= 11:
        #             m.requires_grad = True
        #             print(f'Finetuning layer {name}')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, global_feature, local_feature = self.backbone(x)
        # feature = torch.cat([global_feature, local_feature], dim=1)
        # x = self.mlp(x)
        h = self.hash(global_feature)
        # v = self.variance(local_feature)
        
        h = self.bn_h(h)
        # v = self.bn_v(v)
        
        # v = v / (nn.Tanh()(v * 1))
        h = nn.Tanh()(h * 1) 
        # v = nn.Tanh()(v * 1)
        # v = nn.functional.normalize(v, dim=-1)
        # v = torch.cat([h, v], dim=1)
        
        # x = h * v

        # return x, h, v
        # print("v.shape", v.shape)
        return h, global_feature, local_feature