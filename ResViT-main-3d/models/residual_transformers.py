# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv3d, LayerNorm
from torch.nn.modules.utils import _triple
import torch.nn.functional as F
from scipy import ndimage
from . import transformer_configs as configs


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def posemb_2d_to_3d(posemb, grid):
    """Adapt the pretrained 2D ViT positional table to a 3D token grid.

    The ImageNet checkpoint stores (1, 197, 768): one class token plus a 14x14 grid.
    We drop the class token (this model has none), resize the 14x14 grid to (gh, gw)
    with the same ndimage.zoom the 2D original used, and then replicate it across the
    gd depth positions.

    Replication only sets the INITIAL value. positional_encoding holds one independent
    vector per (z, y, x) token, so every Z position has its own parameters and receives
    its own gradient; the Z positions diverge as soon as training starts.
    """
    gd, gh, gw = grid
    posemb_grid = posemb[0, 1:]                              # drop class token -> (196, C)
    gs_old = int(np.sqrt(posemb_grid.size(0)))
    c = posemb_grid.size(-1)
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, c).numpy()
    zoom = (gh / float(gs_old), gw / float(gs_old), 1)
    posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)    # -> (gh, gw, C)
    posemb_grid = np.broadcast_to(posemb_grid[None], (gd, gh, gw, c))
    return torch.from_numpy(posemb_grid.reshape(1, gd * gh * gw, c).copy())


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size##paraphrase

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3,input_dim=3,old = 1):
        super(Embeddings, self).__init__()
        self.config = config
        img_size = _triple(img_size)
        # The ART block feeds this module a volume already reduced 16x by the CNN encoder
        # (/4) and the ART downsample (/4). The ART skip then restores it with two
        # transposed convs (x4), so the patch stride must be 1 for the shapes to match
        # again -- see the 2D original, where patch_size resolves to 1 for img_size 256.
        # The grid is therefore just the downsampled volume, taken per axis instead of
        # via the 2D-only `img//16//grid` arithmetic.
        patch_size = (1, 1, 1)
        self.grid = (img_size[0] // 16, img_size[1] // 16, img_size[2] // 16)
        n_patches = self.grid[0] * self.grid[1] * self.grid[2]
        in_channels = 1024
        #Learnable patch embeddings
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        #learnable positional encodings (one independent vector per (z,y,x) token)
        self.positional_encoding = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.positional_encoding
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self,config, img_size, vis,in_channels=3,old = 1):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config,img_size=img_size,input_dim=in_channels,old = old)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,dim2=None):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        #use_dropout= use_dropo
        if padding_type == 'reflect':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ART_block(nn.Module):
    def __init__(self,config, input_dim, img_size=224,transformer = None):
        super(ART_block, self).__init__()
        self.transformer = transformer
        self.config = config
        ngf = 64
        mult = 4
        use_bias = False
        norm_layer = nn.BatchNorm3d
        padding_type = 'reflect'
        if self.transformer:
            # Downsample
            model = [nn.Conv3d(ngf * 4, ngf * 8, kernel_size=3,
                               stride=2, padding=1, bias=use_bias),
                     norm_layer(ngf * 8),
                     nn.ReLU(True)]
            model += [nn.Conv3d(ngf * 8, 1024, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(1024),
                      nn.ReLU(True)]
            setattr(self, 'downsample', nn.Sequential(*model))
            #Patch embedings
            self.embeddings = Embeddings(config, img_size=img_size, input_dim=input_dim)
            # Upsampling block
            model = [nn.ConvTranspose3d(self.config.hidden_size, ngf * 8,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                     norm_layer(ngf * 8),
                     nn.ReLU(True)]
            model += [nn.ConvTranspose3d(ngf * 8, ngf * 4,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(ngf * 4),
                      nn.ReLU(True)]
            setattr(self, 'upsample', nn.Sequential(*model))
            #Channel compression
            self.cc = channel_compression(ngf * 8, ngf * 4)
        # Residual CNN
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False,
                             use_bias=use_bias)]
        setattr(self, 'residual_cnn', nn.Sequential(*model))

    def forward(self, x):
        if self.transformer:
            # downsample
            down_sampled = self.downsample(x)
            # embed
            embedding_output = self.embeddings(down_sampled)
            # feed to transformer
            transformer_out, attn_weights = self.transformer(embedding_output)
            B, n_patch, hidden = transformer_out.size()  # reshape from (B, n_patch, hidden) to (B, hidden, d, h, w)
            # The 2D original inferred a square grid with int(np.sqrt(n_patch)); in 3D the
            # grid is generally not cubic, so take the actual (d, h, w) from the embedding.
            d, h, w = self.embeddings.grid
            transformer_out = transformer_out.permute(0, 2, 1)
            transformer_out = transformer_out.contiguous().view(B, hidden, d, h, w)
            # upsample transformer output
            transformer_out = self.upsample(transformer_out)
            # concat transformer output and resnet output
            x = torch.cat([transformer_out, x], dim=1)
            # channel compression
            x = self.cc(x)
        # residual CNN
        x = self.residual_cnn(x)
        return x

########Generator############
class ResViT(nn.Module):
    def __init__(self,config, input_dim, img_size=224, output_dim=3, vis=False):
        super(ResViT, self).__init__()
        self.transformer_encoder = Encoder(config, vis)
        self.config = config
        output_nc = output_dim
        ngf = 64
        use_bias = False
        norm_layer = nn.BatchNorm3d
        padding_type = 'reflect'
        mult = 4

        ############################################################################################
        # Layer1-Encoder1
        model = [nn.ReplicationPad3d(3),
                 nn.Conv3d(input_dim, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        setattr(self, 'encoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer2-Encoder2
        n_downsampling = 2
        model = []
        i = 0
        mult = 2 ** i
        model = [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer3-Encoder3
        model = []
        i = 1
        mult = 2 ** i
        model = [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_3', nn.Sequential(*model))
        ####################################ART Blocks##############################################
        mult = 4
        self.art_1 = ART_block(self.config, input_dim, img_size,transformer = self.transformer_encoder)
        self.art_2 = ART_block(self.config, input_dim, img_size,transformer = None)
        self.art_3 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_4 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_5 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_6 = ART_block(self.config, input_dim, img_size,transformer = self.transformer_encoder)
        self.art_7 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_8 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_9 = ART_block(self.config, input_dim, img_size, transformer=None)
        ############################################################################################
        # Layer13-Decoder1
        n_downsampling = 2
        i = 0
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer14-Decoder2
        i = 1
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer15-Decoder3
        model = []
        model = [nn.ReplicationPad3d(3)]
        model += [nn.Conv3d(ngf, output_dim, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, 'decoder_3', nn.Sequential(*model))

    ############################################################################################
        
    def forward(self, x):
        # Pass input through cnn encoder of ResViT
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)

        #Information Bottleneck
        x = self.art_1(x)
        x = self.art_2(x)
        x = self.art_3(x)
        x = self.art_4(x)
        x = self.art_5(x)
        x = self.art_6(x)
        x = self.art_7(x)
        x = self.art_8(x)
        x = self.art_9(x)

        #decoder
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        return x


    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            if self.config.name == 'b16':
                # embedding/kernel is (1, 1, 1024, 768) -> np2th gives (768, 1024, 1, 1).
                # Our 3D patch embedding is a Conv3d with kernel 1, i.e. weight shape
                # (768, 1024, 1, 1, 1), so the pretrained 1x1 kernel transfers exactly
                # with one extra trailing axis -- no interpolation needed.
                emb_kernel = np2th(weights["embedding/kernel"], conv=True).unsqueeze(-1)
                self.art_1.embeddings.patch_embeddings.weight.copy_(emb_kernel)
                self.art_1.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

                self.art_6.embeddings.patch_embeddings.weight.copy_(emb_kernel)
                self.art_6.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer_encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer_encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            # The 2D original repeated the same 20-line grid-resize block for art_1 and
            # art_6; in 3D both need the same new depth handling, so it lives in one
            # helper (posemb_2d_to_3d) rather than being written out twice.
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            for art in (self.art_1, self.art_6):
                posemb_new = art.embeddings.positional_encoding
                if posemb.size() == posemb_new.size():
                    art.embeddings.positional_encoding.copy_(posemb)
                else:
                    logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                    print('load_pretrained: 2D grid -> 3D grid %s' % (art.embeddings.grid,))
                    art.embeddings.positional_encoding.copy_(
                        posemb_2d_to_3d(posemb, art.embeddings.grid))

            # Encoder whole
            for bname, block in self.transformer_encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)


class Res_CNN(nn.Module):
    def __init__(self, config, input_dim, img_size=224, output_dim=3, vis=False):
        super(Res_CNN, self).__init__()
        self.config = config
        output_nc = output_dim
        ngf = 64
        use_bias = False
        norm_layer = nn.BatchNorm3d
        padding_type = 'reflect'
        mult = 4

        ############################################################################################
        # Layer1-Encoder1
        model = [nn.ReplicationPad3d(3),
                 nn.Conv3d(input_dim, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        setattr(self, 'encoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer2-Encoder2
        n_downsampling = 2
        model = []
        i = 0
        mult = 2 ** i
        model = [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer3-Encoder3
        model = []
        i = 1
        mult = 2 ** i
        model = [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'encoder_3', nn.Sequential(*model))
        ####################################ART Blocks##############################################
        mult = 4
        self.art_1 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_2 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_3 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_4 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_5 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_6 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_7 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_8 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_9 = ART_block(self.config, input_dim, img_size, transformer=None)
        ############################################################################################
        # Layer13-Decoder1
        n_downsampling = 2
        i = 0
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_1', nn.Sequential(*model))
        ############################################################################################
        # Layer14-Decoder2
        i = 1
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'decoder_2', nn.Sequential(*model))
        ############################################################################################
        # Layer15-Decoder3
        model = []
        model = [nn.ReplicationPad3d(3)]
        model += [nn.Conv3d(ngf, output_dim, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, 'decoder_3', nn.Sequential(*model))

    ############################################################################################

    def forward(self, x):
        # Encoder
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)
        # Information bottleneck
        x = self.art_1(x)
        x = self.art_2(x)
        x = self.art_3(x)
        x = self.art_4(x)
        x = self.art_5(x)
        x = self.art_6(x)
        x = self.art_7(x)
        x = self.art_8(x)
        x = self.art_9(x)
        # Decoder
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        return x




class channel_compression(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(channel_compression, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_channels))
        else:
          self.skip = None

        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels))

    def forward(self, x):
        out = self.block(x)
        out += (x if self.skip is None else self.skip(x))
        out = F.relu(out)
        return out
CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-L_16': configs.get_l16_config(),
    'Res-ViT-B_16': configs.get_resvit_b16_config(),
    'Res-ViT-L_16': configs.get_resvit_l16_config(),
}


