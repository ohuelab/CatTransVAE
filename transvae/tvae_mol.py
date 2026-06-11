import os
import json
import random
from time import perf_counter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.set_deterministic_debug_mode("warn")
# torch.autograd.set_detect_anomaly(True) # make training slower


from transvae.tvae_util import *

####### Encoder, Decoder and Generator ############
class EncoderDecoder(nn.Module):
    """
    Base transformer Encoder-Decoder architecture
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        mem, mu, logvar, pred_len = self.encode(src, src_mask)
        x = self.decode(mem, src_mask, tgt, tgt_mask)
        x = self.generator(x)
        return x, mu, logvar, pred_len

    def encode(self, src, src_mask, return_embedded=False):
        return self.encoder(self.src_embed(src), src_mask, return_embedded=return_embedded)

    def decode(self, mem, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), mem, src_mask, tgt_mask)
    
    # def predict_property(self, mu_c):
    #     return self.property_predictor(mu_c)

class Generator(nn.Module):
    "Generates token predictions after final decoder layer"
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab-1)

    def forward(self, x):
        return self.proj(x)
    

class VAEEncoder(nn.Module):
    "Base transformer encoder architecture"
    def __init__(self, layer, N, d_latent, src_len, bypass_bottleneck, eps_scale):
        super().__init__()
        self.layers = clones(layer, N)
        self.conv_bottleneck = ConvBottleneck(layer.size)
        self.z_means, self.z_var = nn.Linear(1984, d_latent), nn.Linear(1984, d_latent) #3648 (512) #576 576
        self.norm = LayerNorm(layer.size)
        self.predict_len1 = nn.Linear(d_latent, d_latent*2)
        self.predict_len2 = nn.Linear(d_latent*2, src_len)
        self.bypass_bottleneck = bypass_bottleneck
        self.eps_scale = eps_scale

    def predict_mask_length(self, mem):
        "Predicts mask length from latent memory so mask can be re-created during inference"
        pred_len = self.predict_len1(mem)
        pred_len = self.predict_len2(pred_len)
        pred_len = F.softmax(pred_len, dim=-1)
        pred_len = torch.topk(pred_len, 1)[1]
        return pred_len

    def reparameterize(self, mu, logvar, eps_scale=1):
        "Stochastic reparameterization"
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) * eps_scale
        return mu + eps*std

    def forward(self, x, mask, return_embedded=False):
        ### Attention and feedforward layers
        for i, attn_layer in enumerate(self.layers):
            x = attn_layer(x, mask)
        ### Batch normalization
        mem = self.norm(x) # (batch, seq_len, d_model)
        embedding = mem.clone()
        ### Convolutional Bottleneck
        if self.bypass_bottleneck:
            mu, logvar = torch.tensor([0.0]), torch.tensor([0.0])
        else:
            mem = mem.permute(0, 2, 1)
            mem = self.conv_bottleneck(mem)
            mem = mem.contiguous().view(mem.size(0), -1)
            # check nan
            assert torch.isfinite(mem).all(), "ENCODER: mem contains inf values after convolutional bottleneck"
            mu, logvar = self.z_means(mem), self.z_var(mem)
            logvar = torch.clamp(logvar, max=10) # prevent overflow in exp
            # check nan
            assert torch.isfinite(mu).all(), "ENCODER: mu contains inf values after linear layer"
            assert torch.isfinite(logvar).all(), "ENCODER: logvar contains inf values after linear layer"
            mem = self.reparameterize(mu, logvar, self.eps_scale)
            # check nan
            assert torch.isfinite(mem).all(), "ENCODER: mem contains inf values after reparameterization"
            pred_len = self.predict_len1(mu)
            pred_len = self.predict_len2(pred_len)
        if return_embedded:
            return mem, mu, logvar, pred_len, embedding
        return mem, mu, logvar, pred_len

class EncoderLayer(nn.Module):
    "Self-attention/feedforward implementation"
    def __init__(self, size, src_len, self_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.src_len = src_len
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 2)

    def forward(self, x, mask, return_attn=False):
        if return_attn:
            attn = self.self_attn(x, x, x, mask, return_attn=True)
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward), attn
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward)

class VAEDecoder(nn.Module):
    "Base transformer decoder architecture"
    def __init__(self, encoder_layers, decoder_layers, N, d_latent, bypass_bottleneck):
        super().__init__()
        self.final_encodes = clones(encoder_layers, 1)
        self.layers = clones(decoder_layers, N)
        self.norm = LayerNorm(decoder_layers.size)
        self.bypass_bottleneck = bypass_bottleneck
        self.size = decoder_layers.size
        self.tgt_len = decoder_layers.tgt_len
        # Reshaping memory with deconvolution
        self.linear = nn.Linear(d_latent, 1984) #3648 (512) #576
        self.deconv_bottleneck = DeconvBottleneck(decoder_layers.size)

    def forward(self, x, mem, src_mask, tgt_mask):
        # check nan
        assert not torch.isnan(mem).any(), "DECODER: mem contains NaN values"
        ### Deconvolutional bottleneck (up-sampling)
        if not self.bypass_bottleneck:
            out = self.linear(mem)
            if torch.isnan(out).any():
                print("NaN detected in linear output")
                print("mem max:", mem.abs().max())
                print("weight max:", self.linear.weight.abs().max())
                raise RuntimeError("Stop here")
            # mem = F.tanh(self.linear(mem))
            mem = F.relu(out)
            # check nan
            assert not torch.isnan(mem).any(), "DECODER: mem contains NaN values after linear transformation"
            mem = mem.view(-1, 64, 31) #57 (3648) #9
            mem = self.deconv_bottleneck(mem)
            mem = mem.permute(0, 2, 1)
        ### Final self-attention layer
        for final_encode in self.final_encodes:
            mem = final_encode(mem, src_mask)
        # Batch normalization
        mem = self.norm(mem)
        # check nan
        assert not torch.isnan(mem).any(), "DECODER: mem contains NaN values after final encoding layers"
        ### Source-attention layers
        for i, attn_layer in enumerate(self.layers):
            x = attn_layer(x, mem, mem, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Self-attention/source-attention/feedforward implementation"
    def __init__(self, size, tgt_len, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.tgt_len = tgt_len
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(self.size, dropout), 3)

    def forward(self, x, memory_key, memory_val, src_mask, tgt_mask, return_attn=False):
        m_key = memory_key
        m_val = memory_val
        if return_attn:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            src_attn = self.src_attn(x, m_key, m_val, src_mask, return_attn=True)
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m_key, m_val, src_mask))
            return self.sublayer[2](x, self.feed_forward), src_attn
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m_key, m_val, src_mask))
            return self.sublayer[2](x, self.feed_forward)

############## Attention and FeedForward ################

class MultiHeadedAttention(nn.Module):
    "Multihead attention implementation (based on Vaswani et al.)"
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads"
        super().__init__()
        assert d_model % h == 0
        #We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, return_attn=False):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        if return_attn:
            return self.attn
        else:
            return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Feedforward implementation"
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


############## BOTTLENECKS #################

class ConvBottleneck(nn.Module):
    """
    Set of convolutional layers to reduce memory matrix to single
    latent vector
    """
    def __init__(self, size):
        super().__init__()
        conv_layers = []
        in_d = size
        first = True
        for i in range(3):
            out_d = int((in_d - 64) // 2 + 64)
            if first:
                kernel_size = 9
                first = False
            else:
                kernel_size = 8
            if i == 2:
                out_d = 64
            conv_layers.append(nn.Sequential(nn.Conv1d(in_d, out_d, kernel_size), nn.MaxPool1d(2)))
            in_d = out_d
        self.conv_layers = ListModule(*conv_layers)

    def forward(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        return x

class DeconvBottleneck(nn.Module):
    """
    Set of deconvolutional layers to reshape latent vector
    back into memory matrix
    """
    # def __init__(self, size):
    #     super().__init__()
    #     deconv_layers = []
    #     in_d = 64
    #     for i in range(3):
    #         out_d = (size - in_d) // 4 + in_d
    #         stride = 3 - i # 3 (3648)
    #         kernel_size = 11 # 10-3*i (3648) # 11
    #         padding = 1 # 4 (3648)
    #         if i == 2:
    #             out_d = size
    #             stride = 2
    #             kernel_size = 11 # 15 (3648) # 11
    #             padding = 1 # 4 (3648)
    #         deconv_layers.append(nn.Sequential(nn.ConvTranspose1d(in_d, out_d, kernel_size,
    #                                                               stride=stride, padding=padding))) # 1
    #         in_d = out_d
    #     self.deconv_layers = ListModule(*deconv_layers)

    def __init__(self, size):
        super().__init__()
        deconv_layers = []
        in_d = 64
        for i in range(3):
            out_d = (size - in_d) // 4 + in_d
            stride = 2
            kernel_size = 11 # 10-3*i (3648) # 11
            padding = 1 # 4 (3648)
            output_padding = 0
            if i == 2:
                out_d = size
                kernel_size = 11 # 15 (3648) # 11
                padding = 1 # 4 (3648)
            if i in (1, 2):
                output_padding = 1
            deconv_layers.append(nn.Sequential(nn.ConvTranspose1d(in_d, out_d, kernel_size,
                                                                      stride=stride, padding=padding,
                                                                      output_padding=output_padding))) # 1
            in_d = out_d
        self.deconv_layers = ListModule(*deconv_layers)

    def forward(self, x):
        for deconv in self.deconv_layers:
            x = F.relu(deconv(x))
        return x

############## Property Predictor #################

class PropertyPredictor(nn.Module):
    "Optional property predictor module"
    def __init__(self, d_pp, depth_pp, d_latent):
        super().__init__()
        prediction_layers = []
        for i in range(depth_pp):
            if i == 0:
                linear_layer = nn.Linear(d_latent, d_pp)
            elif i == depth_pp - 1:
                linear_layer = nn.Linear(d_pp//i, 1)
            else:
                linear_layer = nn.Linear(d_pp//i, d_pp//(i+1))
            prediction_layers.append(linear_layer)
        self.prediction_layers = ListModule(*prediction_layers)

    def forward(self, x):
        for i in range(len(self.prediction_layers)):
            x = torch.tanh(self.prediction_layers[i](x))
            if i != len(self.prediction_layers) - 1:
                x = F.dropout(x, p=0.2, training=self.training)
        x = 50.0 * (x + 1.0)  # Scale to 0-100 range
        return x

############## Embedding Layers ###################

class Embeddings(nn.Module):
    "Transforms input token id tensors to size d_model embeddings"
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Static sinusoidal positional encoding layer"
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

############## Utility Layers ####################

class TorchLayerNorm(nn.Module):
    "Construct a layernorm module (pytorch)"
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.bn = nn.BatchNorm1d(features, eps=eps)

    def forward(self, x):
        return self.bn(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (manual)"
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
        return x + self.dropout(sublayer(self.norm(x)))
    

# class TextConditionEmbedder(nn.Module):
#     """
#     Embedding vector of 768 dimensions to 256 dimensions
#     """
#     def __init__(self, input_dim=768, output_dim=256):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.InstanceNorm1d(input_dim),
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.InstanceNorm1d(512),
#             nn.Linear(512, output_dim),
#         )

#     def forward(self, x):
#         return self.net(x)