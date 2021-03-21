#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART Module.
"""
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Union, List, Optional, Tuple
from parlai.agents.transformer.modules import (TransformerGeneratorModel, 
                                               TransformerEncoderLayer, 
                                               MultiHeadAttention, 
                                               TransformerFFN,)
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.torch import neginf, PipelineHelper
import pdb
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

    APEX_LAYER_NORM = True
except ImportError:
    from torch.nn import LayerNorm

    APEX_LAYER_NORM = False

LAYER_NORM_EPS = 1e-5  # Epsilon for layer norm.


def _normalize(tensor, norm_layer):
    """
    Broadcast layer norm.
    """
    is_cpu = tensor.device == 'cpu' or tensor.device.type == 'cpu'
    if APEX_LAYER_NORM and not is_cpu:
        # fused_layer_norm has a bug around multi-device networks.
        # https://github.com/NVIDIA/apex/issues/770
        # https://github.com/NVIDIA/apex/issues/371
        with torch.cuda.device(tensor.device):
            return norm_layer(tensor)
    else:
        return norm_layer(tensor)

def _create_embeddings(dictionary, embedding_size, padding_idx):
    """
    Create and initialize word embeddings.
    """
    e = nn.Embedding(len(dictionary), embedding_size, padding_idx)
    nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
    nn.init.constant_(e.weight[padding_idx], 0)
    return e

class NarModel(TorchGeneratorModel):
    """
    Non-autoregressive Model.
    """
    @classmethod
    def build_encoder(
        cls,
        opt,
        dictionary,
        embedding=None,
        padding_idx=None,
        reduction_type='mean',
        n_positions=1024,
        n_segments=0,
    ):
        n_layers = (
            opt['n_encoder_layers']
            if opt.get('n_encoder_layers', -1) > 0
            else opt['n_layers']
        )
        return TransformerEncoder(
            n_heads=opt['n_heads'],
            n_layers=n_layers,
            embedding_size=opt['embedding_size'],
            ffn_size=opt['ffn_size'],
            vocabulary_size=len(dictionary),
            embedding=embedding,
            dropout=opt['dropout'],
            attention_dropout=opt['attention_dropout'],
            relu_dropout=opt['relu_dropout'],
            padding_idx=padding_idx,
            learn_positional_embeddings=opt['learn_positional_embeddings'],
            embeddings_scale=opt['embeddings_scale'],
            reduction_type=reduction_type,
            n_positions=n_positions,
            n_segments=n_segments,
            activation=opt['activation'],
            variant=opt['variant'],
            output_scaling=opt['output_scaling'],
        )

    @classmethod
    def build_decoder(
        cls,
        opt,
        dictionary,
        embedding=None,
        padding_idx=None,
        n_positions=1024,
        n_segments=0,
    ):
        n_layers = (
            opt['n_decoder_layers']
            if opt.get('n_decoder_layers', -1) > 0
            else opt['n_layers']
        )
        return TransformerDecoder(
            n_heads=opt['n_heads'],
            n_layers=n_layers,
            embedding_size=opt['embedding_size'],
            ffn_size=opt['ffn_size'],
            vocabulary_size=len(dictionary),
            embedding=embedding,
            dropout=opt['dropout'],
            attention_dropout=opt['attention_dropout'],
            relu_dropout=opt['relu_dropout'],
            padding_idx=padding_idx,
            learn_positional_embeddings=opt['learn_positional_embeddings'],
            embeddings_scale=opt['embeddings_scale'],
            n_positions=n_positions,
            activation=opt['activation'],
            variant=opt['variant'],
            n_segments=n_segments,
        )

    def __init__(self, opt, dictionary):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = dictionary[dictionary.start_token]
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx)
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0,
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024
        n_segments = opt.get('n_segments', 0)

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.encoder = self.build_encoder(
            opt,
            dictionary,
            self.embeddings,
            self.pad_idx,
            reduction_type=None,
            n_positions=n_positions,
            n_segments=n_segments,
        )
        self.decoder = self.build_decoder(
            opt, dictionary, self.embeddings, self.pad_idx, n_positions=n_positions
        )

    def forward(self, *xs, ys=None, prev_enc=None, maxlen=None, bsz=None):
        """
        Get output predictions from the model.

        :param xs:
            input to the encoder
        :type xs:
            LongTensor[bsz, seqlen]
        :param ys:
            Expected output from the decoder. Used
            for teacher forcing to calculate loss.
        :type ys:
            LongTensor[bsz, outlen]
        :param prev_enc:
            if you know you'll pass in the same xs multiple times, you can pass
            in the encoder output from the last forward pass to skip
            recalcuating the same encoder output.
        :param maxlen:
            max number of tokens to decode. if not set, will use the length of
            the longest label this model has seen. ignored when ys is not None.
        :param bsz:
            if ys is not provided, then you must specify the bsz for greedy
            decoding.

        :return:
            (scores, candidate_scores, encoder_states) tuple

            - scores contains the model's predicted token scores.
              (FloatTensor[bsz, seqlen, num_features])
            - candidate_scores are the score the model assigned to each candidate.
              (FloatTensor[bsz, num_cands])
            - encoder_states are the output of model.encoder. Model specific types.
              Feed this back in to skip encoding on the next call.
        """
        assert ys is not None, "Greedy decoding in TGModel.forward no longer supported."
        # TODO: get rid of longest_label
        # keep track of longest label we've ever seen
        # we'll never produce longer ones than that during prediction
        self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        encoder_states = prev_enc if prev_enc is not None else self.encoder(*xs)

        # use teacher forcing
        scores, preds = self.decode_forced(encoder_states, ys)
        return scores, preds, encoder_states

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(
        self, incremental_state: Dict[int, dict], inds: torch.Tensor
    ) -> Dict[int, dict]:
        """
        Reorder the decoder incremental state.

        See ``TorchGeneratorModel.reorder_decoder_incremental_state`` for a description.

        Here, incremental_state is a dict whose keys are layer indices and whose values
        are dicts containing the incremental state for that layer.
        """
        return {
            idx: layer.reorder_incremental_state(incremental_state[idx], inds)
            for idx, layer in enumerate(self.decoder.layers)
        }

    def output(self, tensor):
        """
        Compute output logits.
        """
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        # compatibility with fairseq: fairseq sometimes reuses BOS tokens and
        # we need to force their probability of generation to be 0.
        output[:, :, self.start_idx] = neginf(output.dtype)
        return output


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder layer.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        embeddings_scale=True,
        learn_positional_embeddings=False,
        padding_idx=None,
        n_positions=1024,
        n_segments=0,
        variant='aiayn',
        activation='relu',
        out_dim=None,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.activation = activation
        self.variant = variant

        self.embeddings_scale = embeddings_scale
        self.dropout = nn.Dropout(p=dropout)  # --dropout

        self.n_positions = n_positions
        if out_dim is not None:
            self.out_dim = out_dim
        else:
            self.out_dim = embedding_size

        # self.project_out_layer = None
        # if self.embedding_size != out_dim:
        #     self.project_out_layer = nn.Linear(self.embedding_size, out_dim, bias=False)

        assert (
            embedding_size % n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding

        if (
            self.variant == 'xlm'
            or self.variant == 'prelayernorm'
            or self.variant == 'bart'
        ):
            self.norm_embeddings = LayerNorm(self.dim, eps=LAYER_NORM_EPS)
            if self.variant == 'xlm':
                warn_once(
                    'DEPRECATED: XLM should only be used for backwards compatibility, '
                    'as it involves a less-stable layernorm operation.'
                )
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransformerDecoderLayer(
                    n_heads,
                    embedding_size,
                    ffn_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout,
                    activation=activation,
                    variant=variant,
                )
            )

    def forward_embedding(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ):
        """
        Embed tokens prior to feeding into transformer.

        :param LongTensor[batch, seqlen] input:
            The target input IDs
        :param LongTensor[batch, seqlen] positions:
            Positions for input IDs. If None, computes defaults.
        :param LongTensor[batch, seqlen] segements:
            Segment IDs for extra embedding features. If None, not used.

        :return (tensor, mask):
            embeded input and mask
        """
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        if self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm_embeddings)
        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        if self.variant == 'bart':
            tensor = _normalize(tensor, self.norm_embeddings)

        return tensor

    def forward_layers(
        self,
        tensor: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        incr_state: Dict[int, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of decoder layers.

        :param tensor:
            embedded input tensor for the decoder
        :param enc_out:
            encoder outputs
        :param enc_mask:
            encoder output mask
        :param incr_state:
            Dict mapping layer_idx to incremental state

        :return (tensor, new_incr_state):
            return encoding after applying decoder layers, as well
            as new incremental decoding state.
        """
        new_incr_state = {}
        if getattr(self.layers, 'is_model_parallel', False):
            tensor, new_incr_state = self._apply_model_parallel(
                tensor, encoder_output, encoder_mask, incr_state
            )
        else:
            for idx, layer in enumerate(self.layers):
                tensor, new_incr_state[idx] = layer(
                    x=tensor,
                    encoder_output=encoder_output,
                    encoder_mask=encoder_mask,
                    incr_state=incr_state.get(idx),
                )

        return tensor, new_incr_state

    def forward(self, input, encoder_state, incr_state=None):
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            The incremental state: a dictionary whose keys index the layers and whose
            values contain the incremental state for each layer.
        """
        encoder_output, encoder_mask, pred_len = encoder_state

        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)

        if incr_state is not None:
            # We're doing incremental decoding, so select only the most recent position
            input = input[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        else:
            incr_state = {}

        tensor = self.forward_embedding(input, positions)

        tensor = self.dropout(tensor)  # --dropout

        tensor, new_incr_state = self.forward_layers(
            tensor, encoder_output, encoder_mask, incr_state
        )

        # if self.project_out_layer is not None:
        #     tensor = self.project_out_layer(tensor)

        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm_embeddings)

        return tensor, new_incr_state

    def _apply_model_parallel(self, tensor, encoder_output, encoder_mask, incr_state):
        """
        Pipeline application of model parallelism.
        """
        chunks = PipelineHelper.split(
            (tensor, encoder_output, encoder_mask, incr_state)
        )
        work_items = PipelineHelper.schedule_work_items(self.layers, chunks)

        new_incr_state = {i: [] for i, _ in enumerate(self.layers)}

        for chunk_idx, layer_nos, next_device in work_items:
            s_tensor, s_enc_out, s_enc_mask, s_incr_state = chunks[chunk_idx]
            for layer_no in layer_nos:
                s_tensor, nis = self.layers[layer_no](
                    x=s_tensor,
                    encoder_output=s_enc_out,
                    encoder_mask=s_enc_mask,
                    incr_state=s_incr_state.get(layer_no),
                )
                new_incr_state[layer_no].append(nis)
            # don't move incr state, it's always on the correct device
            s_tensor, s_enc_out, s_enc_mask = PipelineHelper.chunk_to(
                (s_tensor, s_enc_out, s_enc_mask), next_device
            )
            chunks[chunk_idx] = (s_tensor, s_enc_out, s_enc_mask, s_incr_state)

        tensor_out = PipelineHelper.join([c[0] for c in chunks])
        new_incr_state = {
            layer_no: PipelineHelper.join(pieces)
            for layer_no, pieces in new_incr_state.items()
        }

        return tensor_out, new_incr_state

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.n_positions

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        pass

class TransformerDecoderLayer(nn.Module):
    """
    Implements a single Transformer decoder layer.

    Decoder layers are similar to encoder layers but:

    1. Self-attention is limited in a casaul (auto-regressive) manner.
    2. Attend over all of the encoder states.
    """

    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
        activation='relu',
        variant='aiayn',
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.variant = variant
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm1 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.ffn = TransformerFFN(
            embedding_size, ffn_size, relu_dropout=relu_dropout, activation=activation
        )
        self.norm3 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

    def forward(self, x, encoder_output, encoder_mask, incr_state=None):
        """
        Forward pass.

        The incremental state is a dict with values for self- and encoder-attention
        states.
        """

        if incr_state is None:
            incr_state = {}

        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        if self.variant == 'prelayernorm':
            x = _normalize(x, self.norm1)

        # don't peak into the future!
        x, final_self_attn_incr_state = self.self_attention(
            query=x,
            mask=decoder_mask,
            incr_state=incr_state.get('self_attn'),
            static_kv=False,
        )[:2]
        x = self.dropout(x)  # --dropout
        x = x + residual
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = _normalize(x, self.norm1)

        residual = x
        # encoder_attn_layer_norm norm 2
        if self.variant == 'prelayernorm':
            x = _normalize(x, self.norm2)

        x, final_encoder_attn_incr_state = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask,
            incr_state=incr_state.get('encoder_attn'),
            static_kv=True,
        )[:2]
        x = self.dropout(x)  # --dropout
        x = residual + x
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = _normalize(x, self.norm2)

        # finally the ffn
        residual = x
        if self.variant == 'prelayernorm':
            x = _normalize(x, self.norm3)
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = residual + x
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = _normalize(x, self.norm3)

        new_incr_state = {
            'self_attn': final_self_attn_incr_state,
            'encoder_attn': final_encoder_attn_incr_state,
        }
        return x, new_incr_state

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask

    def reorder_incremental_state(
        self, incremental_state: Dict[str, dict], inds: torch.Tensor
    ) -> Dict[str, dict]:
        """
        Reorder all incremental-state tensors for this layer.
        """
        attn_types = {
            'self_attn': self.self_attention,
            'encoder_attn': self.encoder_attention,
        }
        return {
            attn_type: attn.reorder_incremental_state(
                incremental_state[attn_type], inds
            )
            for attn_type, attn in attn_types.items()
        }

class TransformerEncoder(nn.Module):
    """
    Transformer encoder module.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param bool reduction: If true, returns the mean vector for the entire encoding
        sequence.
    :param int n_positions:
        Size of the position embeddings matrix.
    :param int n_segments:
        Number of segments/lang/sentence embeddings.
    :param activation:
        Type of nonlinear activation. Can be relu or gelu.
    :param variant:
        Which transformer architecture to use. Could be AIAYN or XLM.
        Future versions may support things like GPT-2, ...
    :param output_scaling:
        Scale the outputs by a given scalar
    """

    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        padding_idx=0,
        learn_positional_embeddings=False,
        embeddings_scale=False,
        reduction_type='mean',
        n_positions=1024,
        activation='relu',
        variant='aiayn',
        n_segments=0,
        output_scaling=1.0,
        max_target_positions=1024
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.reduction_type = reduction_type
        self.padding_idx = padding_idx
        # this is --dropout, not --relu-dropout or --attention-dropout
        self.dropout_frac = dropout
        self.dropout = nn.Dropout(p=self.dropout_frac)
        self.variant = variant
        self.n_segments = n_segments

        self.n_positions = n_positions
        self.out_dim = embedding_size
        assert (
            embedding_size % n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        # check input formats:
        if embedding is not None:
            assert (
                embedding_size is None or embedding_size == embedding.weight.shape[1]
            ), "Embedding dim must match the embedding size."

        if embedding is not None:
            self.embeddings = embedding
        else:
            raise AssertionError(
                "This code should not execute. Left here in case we want to enable it."
            )
            assert padding_idx is not None
            self.embeddings = nn.Embedding(
                vocabulary_size, embedding_size, padding_idx=padding_idx
            )
            nn.init.normal_(self.embeddings.weight, 0, embedding_size ** -0.5)

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # embedding normalization
        if (
            self.variant == 'xlm'
            or self.variant == 'prelayernorm'
            or self.variant == 'bart'
        ):
            self.norm_embeddings = LayerNorm(self.dim, eps=LAYER_NORM_EPS)
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        if self.n_segments >= 1:
            self.segment_embeddings = nn.Embedding(self.n_segments, self.dim)

        # embed for sequence length
        self.len_embeddings = nn.Embedding(n_positions, embedding_size)
        nn.init.normal_(self.len_embeddings.weight, mean=0, std=0.02)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    n_heads,
                    embedding_size,
                    ffn_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout,
                    variant=variant,
                    activation=activation,
                )
            )
        self.output_scaling = output_scaling

    def forward_embedding(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Embed tokens prior to feeding into transformer.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs
        :param LongTensor[batch,seqlen]:
            If provided, additionally adds ``segments`` as extra embedding features.

        :return (tensor, mask):
            return embedded input and mask
        """
        mask = input != self.padding_idx
        if positions is None:
            positions = (mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)

        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        position_embs = self.position_embeddings(positions).expand_as(tensor)
        tensor = tensor + position_embs

        if self.n_segments >= 1:
            if segments is None:
                segments = torch.zeros_like(input)  # type: ignore
            tensor = tensor + self.segment_embeddings(segments)

        return tensor, mask

    def forward_layers(
        self, tensor: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Apply transformer layers to input.

        :param tensor:
            embedded input
        :param mask:
            mask of input

        :return tensor:
            return embedding after applying transformer layers
        """
        if getattr(self.layers, 'is_model_parallel', False):
            # factored out for readability. It is equivalent to the other
            # condition
            tensor = self._apply_model_parallel(tensor, mask)
        else:
            for i in range(self.n_layers):
                tensor = self.layers[i](tensor, mask)

        return tensor

    def reduce_output(
        self, tensor: torch.Tensor, mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, Optional[torch.BoolTensor]]:
        """
        Reduce transformer output at end of forward pass.

        :param tensor:
            encoded input
        :param mask:
            mask for encoded input

        :return (tensor, mask):
            returns the reduced tensor, and mask if appropriate
        """
        tensor *= self.output_scaling
        if self.reduction_type == 'first':
            return tensor[:, 0, :], None
        elif self.reduction_type == 'max':
            return tensor.max(dim=1)[0], None
        elif self.reduction_type == 'mean':
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1).type_as(tensor)
            output = tensor.sum(dim=1) / divisor
            return output, None
        elif self.reduction_type is None or 'none' in self.reduction_type:
            return tensor, mask
        else:
            raise ValueError(
                "Can't handle --reduction-type {}".format(self.reduction_type)
            )

    def forward(  # type: ignore
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.BoolTensor]]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs
        :param LongTensor[batch,seqlen] segments:
            If provided, additionally adds ``segments`` as extra embedding features.
        """
        # embed input
        tensor, mask = self.forward_embedding(input, positions, segments)

        if self.variant == 'xlm' or self.variant == 'bart':
            tensor = _normalize(tensor, self.norm_embeddings)

        # add param for predicting target length
        len_tokens = self.len_embeddings(input.new(input.size(0), 1).fill_(0))
        tensor = torch.cat([len_tokens, tensor], dim=1)

        # add mask for predicting target length
        mask = torch.cat([mask.new(input.size(0), 1).fill_(0), mask], dim=1)
        if not mask.any():
            mask = None

        # pdb.set_trace()
        # --dropout on the embeddings
        tensor = self.dropout(tensor)

        tensor *= mask.unsqueeze(-1).type_as(tensor)

        # apply transformer layers
        tensor = self.forward_layers(tensor, mask)

        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm_embeddings)

        # predict length of decoding sequence    
        # pdb.set_trace()    
        pred_len_logits = torch.matmul(tensor.clone()[:, 0, :], self.len_embeddings.weight).float()
        # pred_len_logits[:, 0] += float('-inf')   # Cannot predict the len_token
        pred_len = F.log_softmax(pred_len_logits, dim=-1)

        # pdb.set_trace()
        # restore logits and mask
        tensor = tensor[:, 1:, :]
        if mask is not None:
            mask = mask[:, 1:]


        # reduce output
        tensor, out_mask = self.reduce_output(tensor, mask)
        if out_mask is not None:
            return tensor, out_mask, pred_len
        else:
            return tensor, pred_len

    def _apply_model_parallel(self, tensor, mask):
        """
        Pipeline application of model parallelism.
        """
        chunks = PipelineHelper.split((tensor, mask))
        work_items = PipelineHelper.schedule_work_items(self.layers, chunks)

        for chunk_idx, layer_nos, next_device in work_items:
            s_tensor, s_mask = chunks[chunk_idx]
            for layer_no in layer_nos:
                s_tensor = self.layers[layer_no](s_tensor, s_mask)
            chunks[chunk_idx] = PipelineHelper.chunk_to((s_tensor, s_mask), next_device)

        tensor_out, mask_out = PipelineHelper.join(chunks)
        return tensor_out


    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.n_positions

class TransformerEncoderLayer(nn.Module):
    """
    Implements a single Transformer encoder layer.
    """

    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
        activation='relu',
        variant=None,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.activation = activation
        self.variant = variant
        self.attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout  # --attention-dropout
        )
        self.norm1 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.ffn = TransformerFFN(
            embedding_size,
            ffn_size,
            relu_dropout=relu_dropout,
            activation=self.activation,
        )
        self.norm2 = LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor, mask):
        """
        Forward pass.
        """
        residual = tensor
        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm1)

        attended_tensor = self.attention(tensor, mask=mask)[0]
        tensor = residual + self.dropout(attended_tensor)

        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            tensor = _normalize(tensor, self.norm1)

        residual = tensor

        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm2)

        tensor = residual + self.dropout(self.ffn(tensor))

        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            tensor = _normalize(tensor, self.norm2)
            
        tensor *= mask.unsqueeze(-1).type_as(tensor)
        return tensor


