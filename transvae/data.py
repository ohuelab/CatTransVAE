import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from rdkit import Chem
import torch
from torch.autograd import Variable
from transvae.tvae_util import *
from torch.utils.data import Dataset
    
class MolDataset(Dataset):
    def __init__(self, data_idx, data_mol):
        self.data_idx = data_idx
        self.data_mol = data_mol

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        # Return a tuple: data_idx, data_mol
        return (self.data_idx[idx],
                self.data_mol[idx])

    
def vae_data_gen(data_source, idx, mols, char_dict):
    encoding_len = 300
    smiles = mols
    del mols
    smiles = [tokenizer(x) for x in smiles]
    # limit the length
    smiles = [smi[:encoding_len-2] if len(smi) > encoding_len-2 else smi for smi in smiles]
    encoded_data = torch.empty((len(smiles), encoding_len)) 
    for j, smi in tqdm(enumerate(smiles), desc="Encoding mol SMILES"):
        encoded_smi = encode_smiles(smi, encoding_len-1, char_dict) #-start
        encoded_smi = [0] + encoded_smi
        encoded_data[j,:] = torch.tensor(encoded_smi)
    idx = np.array([str(i) for i in idx])
    return idx, encoded_data


def make_std_mask(tgt, pad):
    """
    Creates sequential mask matrix for target input (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)

    Arguments:
        tgt (torch.tensor, req): Target vector of token ids
        pad (int, req): Padding token id
    Returns:
        tgt_mask (torch.tensor): Sequential target mask
    """
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask

