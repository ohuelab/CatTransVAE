#ref: https://github.com/MolecularAI/Chemformer/blob/53a2819076bd16f36131839c7fb88157cfc2ce92/molbart/utils/samplers/beam_search_samplers.py

from __future__ import annotations
import numpy as np
import torch
from rdkit import Chem, RDLogger
from typing import Any, Dict, List, Tuple

RDLogger.DisableLog("rdApp.*")

import torch
import torch.utils.data as tud
from transvae.tvae_util import *

bad_token_ll = -1e5
length_norm=None

class Node:
    def __init__(self, model, x, vocabulary, device, data_device="cpu", batch_size=64):
        """
        Initialize a Node used for autoregression
        predictions, such as greedy search, multinomial
        sampling, or beam search

        Parameters:
            model (Any): any autoregressive model
            x (tuple(torch.tensor,)): a torch tensor representing
                              additional data to pass to
                              the regression model
            vocabulary (Vocabulary): a vocabulary object
            device (torch.device, str): device where to place
                                        the model
            data_device (torch.device, str): device where to place
                                             the data. WARNING! Use
                                             gpu here sparingly.
            batch_size(int): internal batch size used for the beam search
                             (Default: 64)

        """
        assert isinstance(device, torch.device) or isinstance(device, str)
        assert isinstance(data_device, torch.device) or isinstance(data_device, str)

        if isinstance(device, str):
            device = torch.device(device)

        if isinstance(data_device, str):
            data_device = torch.device(data_device)

        self.model = model
        self.device = device
        self.data_device = data_device

        src = x["encoder_input"]
        src_mask = x["encoder_pad_mask"]
        self.batch_size = batch_size  # min(batch_size, len(src))

        with torch.no_grad():
            self.model = self.model.eval()

            if next(self.model.parameters()).device != self.device:
                self.model = self.model.to(self.device)
            if src.device != self.device:
                src = src.to(self.device)
            if src_mask.device != self.device:
                src_mask = src_mask.to(self.device)

            self.x = self.model.encode(x).detach().permute(1, 0, 2)
            self.x_mask = src_mask.detach().transpose(0, 1)

            if self.x.device != self.data_device:
                self.x = self.x.to(self.data_device)
            if self.x_mask.device != self.data_device:
                self.x_mask = self.x_mask.to(self.data_device)

        self.vocabulary = vocabulary

        self.y = torch.ones((self.x.shape[0], 1), dtype=torch.long) * self.vocabulary["start"]
        self.y = self.y.detach()

        if self.y.device != self.data_device:
            self.y = self.y.to(self.data_device)

        self.ll_mask = torch.tensor([False])
        self.pos = 0

    def set_beam_width(self, beam_width):
        self.beam_width = beam_width

    def _get_topk(self, loglikelihood):
        v = loglikelihood.shape[-1]
        loglikelihood, next_chars = loglikelihood.topk(k=min(v, self.beam_width), axis=-1)
        if v < self.beam_width:
            d = self.beam_width - len(self.vocabulary)
            pl = -1e20 * torch.ones(
                (len(loglikelihood), d),
                dtype=loglikelihood.dtype,
                device=loglikelihood.device,
            )
            pc = torch.zeros(
                (len(next_chars), d),
                dtype=next_chars.dtype,
                device=loglikelihood.device,
            )
            loglikelihood = torch.cat((loglikelihood, pl), dim=-1)
            next_chars = torch.cat((next_chars, pc), dim=-1)
        return loglikelihood, next_chars

    def _init_action(self, loglikelihood):
        # Perform the first step
        loglikelihood, next_chars = self._get_topk(loglikelihood)

        self.loglikelihood = loglikelihood.view(-1, 1)
        next_chars = next_chars.view(-1, 1)

        self.y = self.y.view(len(self.y), 1, -1).repeat(1, self.beam_width, 1).view(-1, 1)
        self.x = self.x[:, None].repeat(1, self.beam_width, 1, 1).view((-1,) + tuple(self.x.shape[1:]))
        self.x_mask = self.x_mask[:, None].repeat(1, self.beam_width, 1).view((-1,) + tuple(self.x_mask.shape[1:]))

        self.y = torch.cat((self.y, next_chars), dim=-1)

        # VERY IMPORTANT! we need a mask for
        # the log likelihood when reaching the eos
        # self.ll_mask = torch.zeros(len(self.loglikelihood), dtype=torch.bool)
        self.ll_mask = torch.any(self.y == self.vocabulary["end"], dim=-1)

    def get_actions(self):
        batch_size = self.batch_size
        next_loglikelihood = []

        local_dataset = tud.TensorDataset(self.x, self.x_mask, self.y)
        local_loader = tud.DataLoader(local_dataset, batch_size=batch_size)

        # make sure that the local_loader
        # will be iterated over only once
        iterator = iter(local_loader)

        with torch.no_grad():
            for x, x_mask, y in local_loader:
                if x.device != self.device:
                    x = x.to(self.device)
                if x_mask.device != self.device:
                    x_mask = x_mask.to(self.device)
                if y.device != self.device:
                    y = y.to(self.device)

                X = {"decoder_input": y, "memory_input": x, "memory_pad_mask": x_mask}

                ll = self.model.decode_batch(X)
                next_loglikelihood.append(ll)
        next_loglikelihood = torch.cat(next_loglikelihood, axis=0)
        next_loglikelihood = next_loglikelihood.detach()
        if next_loglikelihood != self.data_device:
            next_loglikelihood = next_loglikelihood.to(self.data_device)

        return next_loglikelihood

    def action(self, next_loglikelhihood):
        if self.pos == 0:
            self._init_action(next_loglikelhihood)
        else:
            vocabulary_size = len(self.vocabulary)
            # set loglikehihood to the maxium (0)
            # when observed an eos_token
            next_loglikelhihood[self.ll_mask, :] = (
                torch.minimum(self.loglikelihood.min(), next_loglikelhihood.min()) - 1.0
            )
            next_loglikelhihood[self.ll_mask, self.vocabulary["end"]] = 0.0
            # done

            ll = (self.loglikelihood + next_loglikelhihood).view(-1, self.beam_width, vocabulary_size)
            ll, idx = self._get_topk(ll.flatten(start_dim=1))

            # tricky indexing
            next_chars = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()
            if best_candidates.device != self.device:
                best_candidates = best_candidates.to(self.device)
            # done

            y = self.y.view(-1, self.beam_width, self.y.shape[-1])
            i = torch.arange(len(y)).unsqueeze(-1).repeat(1, self.beam_width).flatten()
            j = best_candidates.flatten()
            self.y = y[i, j].view(-1, self.y.shape[-1])

            self.y = torch.cat((self.y, next_chars), dim=-1)
            self.loglikelihood = ll.view(-1, 1)

            # update ll mask
            self.ll_mask = torch.any(self.y == self.vocabulary["end"], dim=-1)
        self.pos = self.pos + 1

    @staticmethod
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        A = subsequent_mask == 0
        return A.type(torch.long)


class Criterion:
    def __call__(self, node):
        raise NotImplementedError("Not implemented")


class MaxLength(Criterion):
    def __init__(self, max_length):
        super(MaxLength, self).__init__()
        self.max_length = max_length

    def __call__(self, node):
        return node.pos >= self.max_length - 1


class EOS(Criterion):
    def __init__(self):
        super(EOS, self).__init__()

    def __call__(self, node):
        return torch.all(node.ll_mask).item()


class LogicalAnd(Criterion):
    def __init__(self, criteria):
        super(LogicalAnd, self).__init__()
        self.criteria = criteria

    def __call__(self, node):
        return all([c(node) for c in self.criteria])


class LogicalOr(Criterion):
    def __init__(self, criteria):
        super(LogicalOr, self).__init__()
        self.criteria = criteria

    def __call__(self, node):
        return any([c(node) for c in self.criteria])


def beamsearch(node, beamsize, stop_criterion):
    node.set_beam_width(beamsize)
    print("Sampling with beam size: " + str(beamsize))

    while not stop_criterion(node):
        a = node.get_actions()
        node.action(a)

    a = node.get_actions()

    end_tokens = node.vocabulary["end"] * torch.logical_not(node.ll_mask).type(node.y.dtype)
    node.y = torch.cat((node.y, end_tokens.view(-1, 1)), dim=-1)
    ll_tail = a[torch.arange(len(a)), end_tokens] * torch.logical_not(node.ll_mask).type(a.dtype)
    node.loglikelihood = node.loglikelihood + ll_tail.view(-1, 1)
    return node

def _update_beams_(i, decode_fn, token_ids_list, pad_mask_list, lls_list, pad_token_id=0, end_token_id=1, max_seq_len=300):
    """Update beam tokens and pad mask in-place using a single decode step

    Updates token ids and pad mask in-place by producing the probability distribution over next tokens
    and choosing the top k (number of beams) log likelihoods to choose the next tokens.
    Sampling is complete if every batch element in every beam has produced an end token.

    Args:
        i (int): The current iteration counter
        decode_fn (fn): Function used to apply tokens to model and produce log probability distribution
        token_ids_list (List[torch.Tensor]): List of token_ids, each of shape [seq_len, batch_size]
        pad_mask_list (List[torch.Tensor]): List of pad_masks, each of shape [seq_len, batch_size]
        lls_list (List[torch.Tensor]): List of log likelihoods, each of shape [batch_size]

    Returns:
        (bool): Specifies whether all of the beams are complete
    """

    assert len(token_ids_list) == len(pad_mask_list) == len(lls_list)

    num_beams = len(token_ids_list)

    ts = [token_ids[:i, :] for token_ids in token_ids_list]
    ms = [pad_mask[:i, :] for pad_mask in pad_mask_list]
    ms = [subsequent_mask(t.size(0)).long().to(ts[0].device) for t in ts]
    # print("ts:", [t.shape for t in ts])
    # print("ms:", [m.shape for m in ms])

    # Apply current seqs to model to get a distribution over next tokens
    # new_lls is a tensor of shape [batch_size, vocab_size * num_beams]
    new_lls = [_beam_step(decode_fn, t, m, lls, pad_token_id=pad_token_id, end_token_id=end_token_id) for t, m, lls in zip(ts, ms, lls_list)]
    # print("new_lls:", [new_ll for new_ll in new_lls])
    norm_lls = [_norm_length(lls, mask) for lls, mask in zip(new_lls, ms)]
    # print("norm_lls:", [norm_ll for norm_ll in norm_lls])

    _, vocab_size = tuple(norm_lls[0].shape)
    new_lls = torch.cat(new_lls, dim=1)
    norm_lls = torch.cat(norm_lls, dim=1)

    # Keep lists (of length num_beams) of tensors of shape [batch_size]
    top_lls, top_idxs = torch.topk(norm_lls, num_beams, dim=1)
    new_ids_list = list((top_idxs % vocab_size).T)
    new_ids_list = [new_ids+1 for new_ids in new_ids_list] # add 1
    beam_idxs_list = list((top_idxs // vocab_size).T)
    top_lls = [new_lls[b_idx, idx] for b_idx, idx in enumerate(list(top_idxs))]
    top_lls = torch.stack(top_lls).T
    # print("top_lls:", [top_ll for top_ll in top_lls])

    beam_complete = []
    new_ts_list = []
    new_pm_list = []
    new_lls_list = []

    # Set the sampled tokens, pad masks and log likelihoods for each of the new beams
    for new_beam_idx, (new_ids, beam_idxs, lls) in enumerate(zip(new_ids_list, beam_idxs_list, top_lls)):
        # Get the previous sequences corresponding to the new beams
        token_ids = [token_ids_list[beam_idx][:, b_idx] for b_idx, beam_idx in enumerate(beam_idxs)]
        token_ids = torch.stack(token_ids).transpose(0, 1)

        # Generate next elements in the pad mask. An element is padded if:
        # 1. The previous token is an end token
        # 2. The previous token is a pad token
        is_end_token = token_ids[i - 1, :] == end_token_id
        is_pad_token = token_ids[i - 1, :] == pad_token_id
        new_pad_mask = torch.logical_or(is_end_token, is_pad_token)
        beam_complete.append(new_pad_mask.sum().item() == new_pad_mask.numel())

        # Ensure all sequences contain an end token
        if i == max_seq_len - 1:
            new_ids[~new_pad_mask] = end_token_id

        # Set the tokens to pad if an end token as already been produced
        new_ids[new_pad_mask] = pad_token_id
        token_ids[i, :] = new_ids

        # Generate full pad mask sequence for new token sequence
        pad_mask = [pad_mask_list[beam_idx][:, b_idx] for b_idx, beam_idx in enumerate(beam_idxs)]
        pad_mask = torch.stack(pad_mask).transpose(0, 1)
        pad_mask[i, :] = new_pad_mask

        # Add tokens, pad mask and lls to list to be updated after all beams have been processed
        new_ts_list.append(token_ids)
        new_pm_list.append(pad_mask)
        new_lls_list.append(lls)

    complete = sum(beam_complete) == len(beam_complete)

    # Update all tokens, pad masks and lls
    if not complete:
        for beam_idx, (ts, pm, lls) in enumerate(zip(new_ts_list, new_pm_list, new_lls_list)):
            token_ids_list[beam_idx] = ts
            pad_mask_list[beam_idx] = pm
            lls_list[beam_idx] = lls

    return complete

def _beam_step(decode_fn, tokens, mask, lls, pad_token_id=0, end_token_id=1):
    """Apply tokens to model to produce the log likelihoods for the full sequence

    A single iteration of decode is applied to the model to produce the next tokens in the sequences
    and the log likelihoods for the entire sequences (including the next token)
    The lls are returned as a distribution over all possible next tokens

    Args:
        decode_fn (fn): Function used to apply tokens to model and produce log probability distribution
        tokens (torch.Tensor): Tensor of shape [seq_len, batch_size] containing the current token ids
        mask (torch.Tensor): BoolTensor of shape [seq_len, batch_size] containing the padding mask
        lls (torch.Tensor): Tensor of shape [batch_size] containing log likelihoods for seqs so far

    Returns:
        seq_lls (torch.Tensor): Tensor of shape [batch_size, vocab_size]
    """
    output, logits, prob, log_probs = decode_fn(target=tokens.T, target_pad_mask=mask)
    next_token_lls = log_probs
    
    # output_dist = decode_fn(tokens.T, mask.T)
    # next_token_lls = output_dist[-1, :, :].cpu()

    # Create a vector from which only a pad token can be sampled
    _, vocab_size = tuple(next_token_lls.shape)
    complete_seq_ll = torch.ones((1, vocab_size), device=next_token_lls.device) * bad_token_ll
    complete_seq_ll[:, pad_token_id] = 0.0

    # Use this vector in the output for sequences which are complete
    is_end_token = tokens[-1, :] == end_token_id
    is_pad_token = tokens[-1, :] == pad_token_id
    # ll_mask = torch.logical_or(is_end_token, is_pad_token).cpu().unsqueeze(1)
    ll_mask = torch.logical_or(is_end_token, is_pad_token).unsqueeze(1).to(next_token_lls.device)
    masked_lls = (ll_mask * complete_seq_ll) + (~ll_mask * next_token_lls)

    seq_lls = (lls + masked_lls.T).T
    return seq_lls

def _norm_length(seq_lls, mask):
    """Normalise log-likelihoods using the length of the constructed sequence
    Equation from:
    Wu, Yonghui, et al.
    "Google's neural machine translation system: Bridging the gap between human and machine translation."
    arXiv preprint arXiv:1609.08144 (2016).

    Args:
        seq_lls (torch.Tensor): Tensor of shape [batch_size, vocab_size] containing log likelihoods for seqs so far
        mask (torch.Tensor): BoolTensor of shape [seq_len, batch_size] containing the padding mask

    Returns:
        norm_lls (torch.Tensor): Tensor of shape [batch_size, vocab_size]
    """

    if length_norm is not None:
        seq_lengths = (~mask).sum(dim=0)
        norm = torch.pow(5 + seq_lengths, length_norm) / pow(6, length_norm)
        norm_lls = (seq_lls.T / norm.cpu()).T
        return norm_lls

    return seq_lls

# @staticmethod
def _transpose_list(list_):
    """Transpose 2D list so that inner dimension is first

    Args:
        l (List[Any]): List to be transposed

    Returns:
        (List[Any]): Transposed list
    """

    outer_dim = len(list_)
    inner_dim = len(list_[0])

    transposed = [[[]] * outer_dim for _ in range(inner_dim)]
    for outer_idx, inner in enumerate(list_):
        for inner_idx, item in enumerate(inner):
            transposed[inner_idx][outer_idx] = item

    return transposed

# @staticmethod
def _sort_beams(mol_strs, log_lhs):
    """Return mols sorted by their log likelihood

    Args:
        mol_strs (List[List[str]]): SMILES encoding of molecules
        log_lhs (List[List[float]]): Log likelihood for each molecule

    Returns:
        (List[str], List[float]): Tuple of sorted molecules and sorted log lhs
    """

    assert len(mol_strs) == len(log_lhs)

    sorted_mols = []
    sorted_lls = []

    for mols, lls in zip(mol_strs, log_lhs):
        mol_lls = sorted(zip(mols, lls), reverse=True, key=lambda mol_ll: mol_ll[1])
        mols, lls = tuple(zip(*mol_lls))
        sorted_mols.append(list(mols))
        sorted_lls.append(list(lls))

    return sorted_mols, sorted_lls