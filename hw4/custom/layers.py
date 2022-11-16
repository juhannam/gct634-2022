import utils

import math as m
import numpy as np
import math
import torch
import torch.nn.functional as F


def sinusoid(max_seq, embedding_dim):
    return np.array([[
        [
            m.sin(
                pos * m.exp(-m.log(10000) * i / embedding_dim) * m.exp(
                    m.log(10000) / embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
            )
            for i in range(embedding_dim)
        ]
        for pos in range(max_seq)
    ]])


class DynamicPositionEmbedding(torch.nn.Module):
    '''
    Applies sinusoidal positional embedding to input

    '''
    def __init__(self, embedding_dim, max_seq=2048):
        super().__init__()
        embed_sinusoid_list = np.array([[
            [
                m.sin(
                    pos * m.exp(-m.log(10000) * i/embedding_dim) *
                    m.exp(m.log(10000)/embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
                )
                for i in range(embedding_dim)
            ]
            for pos in range(max_seq)
        ]])
        self.positional_embedding = embed_sinusoid_list

    def forward(self, x):
        x = x + torch.from_numpy(self.positional_embedding[:, :x.size(1), :]).to(x.device, dtype=x.dtype)
        return x


class RelativeGlobalAttention(torch.nn.Module):
    """
    from Music Transformer ( Huang et al, 2018 )
    [paper link](https://arxiv.org/pdf/1809.04281.pdf)
    """
    def __init__(self, h=4, d=256, add_emb=False, max_seq=2048, **kwargs):
        super().__init__()
        self.len_k = None
        self.max_seq = max_seq
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = torch.nn.Linear(self.d, self.d)
        self.Wk = torch.nn.Linear(self.d, self.d)
        self.Wv = torch.nn.Linear(self.d, self.d)
        self.fc = torch.nn.Linear(d, d)

        ##### You don't need to know this ###
        self.additional = add_emb
        self.E = torch.nn.Parameter(torch.randn([self.max_seq, int(self.dh)], requires_grad=True))
        #####

    def forward(self, inputs, mask=None, **kwargs):
        """
        inputs: a list of tensors. i.e) [Q, K, V]
        mask: mask tensor

        return: final tensor ( output of attention )
        """
        ############################################################################
        ### Attention! ###

        ##### Obtain query tensor #####
        q = inputs[0]
        q = self.Wq(q)
        q = torch.reshape(q, (q.size(0), q.size(1), self.h, -1))
        q = q.permute(0, 2, 1, 3)  # batch, h, seq, dh

        ##### Obtain key tensor #####
        k = inputs[1]
        k = self.Wk(k)
        k = torch.reshape(k, (k.size(0), k.size(1), self.h, -1))
        k = k.permute(0, 2, 1, 3)

        ##### Obtain value tensor #####
        v = inputs[2]
        v = self.Wv(v)
        v = torch.reshape(v, (v.size(0), v.size(1), self.h, -1))
        v = v.permute(0, 2, 1, 3)

        ##### This part is for obtaining relative logits for relative global attention. #####
        self.len_k = k.size(2)
        self.len_q = q.size(2)
        E = self._get_left_embedding(self.len_q, self.len_k)
        QE = torch.einsum('bhld,md->bhlm', [q, E])
        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        ##### Obtain QKt #####
        Kt = k.permute(0, 1, 3, 2)
        QKt = torch.matmul(q, Kt) # batch, h, seq, seq

        ##### Add relative logits to QKt, and divide by dh #####
        logits = QKt + Srel
        logits = logits / math.sqrt(self.dh)

        ##### Apply masking in order to prevent the query seeing future keys #####
        if mask is not None:
            logits += (mask.to(torch.int64) * -1e9).to(logits.dtype)

        ##### Apply softmax to each row, and multiply to value tensor. #####
        attention_weights = F.softmax(logits, -1)
        attention = torch.matmul(attention_weights, v)

        out = attention.permute(0, 2, 1, 3) # batch, seq, h, dh

        ##### Reshape the output so that the multi heads are concatenated back together.
        out = torch.reshape(out, (out.size(0), -1, self.d))

        ##### last fc layer to project the concatenated output #####
        out = self.fc(out)
        return out, attention_weights

    ############################################################################
    ##### These 3 functions below are related to relative global attention #####

    def _get_left_embedding(self, len_q, len_k):
        starting_point = max(0,self.max_seq-len_q)
        e = self.E[starting_point:,:]
        return e

    def _skewing(self, tensor: torch.Tensor):
        padded = F.pad(tensor, [1, 0, 0, 0, 0, 0, 0, 0])
        reshaped = torch.reshape(padded, shape=[padded.size(0), padded.size(1), padded.size(-1), padded.size(-2)])
        Srel = reshaped[:, :, 1:, :]
        if self.len_k > self.len_q:
            Srel = F.pad(Srel, [0, 0, 0, 0, 0, 0, 0, self.len_k-self.len_q])
        elif self.len_k < self.len_q:
            Srel = Srel[:, :, :, :self.len_k]

        return Srel

    @staticmethod
    def _qe_masking(qe):
        mask = utils.sequence_mask(
            torch.arange(qe.size()[-1] - 1, qe.size()[-1] - qe.size()[-2] - 1, -1).to(qe.device),
            qe.size()[-1])
        mask = ~mask.to(mask.device)
        return mask.to(qe.dtype) * qe

    ############################################################################


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=2048):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.rga = RelativeGlobalAttention(h=h, d=d_model, max_seq=max_seq, add_emb=additional)

        self.FFN_pre = torch.nn.Linear(self.d_model, self.d_model//2)
        self.FFN_suf = torch.nn.Linear(self.d_model//2, self.d_model)

        self.layernorm1 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    def forward(self, x, mask=None, **kwargs):
        attn_out, w = self.rga([x,x,x], mask)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out+x)

        ffn_out = F.relu(self.FFN_pre(out1))
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layernorm2(out1+ffn_out)
        return out2, w



class Encoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, h, input_vocab_size, rate=0.1, max_len=None):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=d_model)
        if True:
            self.pos_encoding = DynamicPositionEmbedding(self.d_model, max_seq=max_len)

        self.enc_layers = torch.nn.ModuleList(
            [EncoderLayer(d_model, rate, h=h, additional=False, max_seq=max_len)
             for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(rate)

    def forward(self, x, mask=None):
        weights = []
        # adding embedding and position encoding.
        x = self.embedding(x.to(torch.long))  # (batch_size, input_seq_len, d_model)
        x *= math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x, w = self.enc_layers[i](x, mask)
            weights.append(w)
        return x, weights # (batch_size, input_seq_len, d_model)
