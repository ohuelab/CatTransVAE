import re
import math
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import ChiralType
fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)

rdBase.DisableLog('rdApp.*')

import seaborn as sns

# Set font to Helvetica
# plt.rcParams["font.sans-serif"] = ["Helvetica"]
plt.rcParams["font.family"] = "sans-serif"

# Alternatively, set directly in sns.set_theme
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(font="Helvetica", style="ticks", rc=custom_params)
sns.set_theme(style="ticks", rc=custom_params)

# COLOR_MAIN = "#F34D6F"
COLOR_MAIN = "#E37185"
COLOR_COMPARE = "#595A5A"
# COLOR_SET = ["#5971C5", "#F29344", "#F34D6F", "#458B73", "#F1C54E", "#7955BD", "#5DA6C5", "#96606B", "#ADCA63", "#AE2A2A", "#7D7330"]
COLOR_SET = ["#5971C5", "#F29344", "#E37185", "#458B73", "#F1C54E", "#7955BD", "#5DA6C5", "#96606B", "#ADCA63", "#AE2A2A", "#7D7330"]
COLOR_BLACK = "#595A5A"


######## MODEL HELPERS ##########

def clones(module, N):
    """Produce N identical layers (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    """Mask out subsequent positions (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)"""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention' (adapted from Viswani et al.)"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class ListModule(nn.Module):
    """Create single pytorch module from list of modules"""
    def __init__(self, *args):
        super().__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class KLAnnealer:
    """
    Scales KL weight (beta) linearly according to the number of epochs
    """
    def __init__(self, kl_low, kl_high, n_epochs, start_epoch):
        self.kl_low = kl_low
        self.kl_high = kl_high
        self.n_epochs = n_epochs
        self.start_epoch = start_epoch

        self.kl = (self.kl_high - self.kl_low) / (self.n_epochs - self.start_epoch)

    def __call__(self, epoch):
        k = (epoch - self.start_epoch) if epoch >= self.start_epoch else 0
        beta = self.kl_low + k * self.kl
        if beta > self.kl_high:
            beta = self.kl_high
        else:
            pass
        return beta
    
    # for save state dict
    def state_dict(self):
        return {
            'kl_low': self.kl_low,
            'kl_high': self.kl_high,
            'n_epochs': self.n_epochs,
            'start_epoch': self.start_epoch,
            'kl': self.kl
        }
    
    def load_state_dict(self, state_dict):
        if state_dict is not None:
            self.kl_low = state_dict['kl_low']
            self.kl_high = state_dict['kl_high']
            self.n_epochs = state_dict['n_epochs']
            self.start_epoch = state_dict['start_epoch']
            self.kl = state_dict['kl']

####### MOLECULE HELPERS ##########

# SMILES to mol
def mol_to_smiles(mol):
    smiles = Chem.MolToSmiles(mol)
    try:
        smiles = Chem.CanonSmiles(smiles)
    except:
        print("[mol_to_smiles] Invalid SMILES: %s" % smiles)
        return None
    return smiles

# mol to SMILES
def smiles_to_mol(smiles):
    try:
        smiles = Chem.CanonSmiles(smiles)
    except:
            
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            print("[smiles_to_mol] Invalid SMILES: %s" % smiles)
            return None
    return mol

# normalize SMILES
def normalize_smiles(smiles):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return mol_to_smiles(mol)

# cleaner smiles
def cleaner(smiles, max_len):
    if pd.isna(smiles) or str(smiles).strip() == '' or str(smiles).strip().lower() == 'nan':
        print("[cleaner] SMILES is NaN or empty: %s" % smiles)
        return None
    smiles = str(smiles).strip().split('\n')[0]
    if len(smiles) > max_len * 2:
        print("[cleaner] SMILES too long: %s" % smiles)
        return None
    smiles = normalize_smiles(smiles)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("[cleaner] Invalid SMILES: %s" % smiles)
            return None
        # remove isotopes
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
        # remove some chiral tags except tetrahedral 
        for atom in mol.GetAtoms():
            if atom.GetChiralTag() in (
                ChiralType.CHI_SQUAREPLANAR,
                ChiralType.CHI_TRIGONALBIPYRAMIDAL,
                ChiralType.CHI_OCTAHEDRAL,
                ChiralType.CHI_OTHER,
                ChiralType.CHI_ALLENE
            ):
                atom.SetChiralTag(ChiralType.CHI_UNSPECIFIED)
        smiles = Chem.MolToSmiles(mol)
    except Exception as e:
        print("[cleaner] Error processing SMILES: %s, error: %s" % (smiles, str(e)))
        return None
    token = tokenizer(smiles)
    if len(token) >= max_len-2: # account for start and end tokens
        print("[cleaner] SMILES too long after tokenization: %s" % smiles)
        return None
    return smiles

# cleaner for kekulization issues
def clean_kekulize(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is not None:
        count_while = 0
        def is_roundtrip_valid(mol):
            if mol is None:
                return False
            try:
                smiles = Chem.MolToSmiles(mol)
                return Chem.MolFromSmiles(smiles) is not None
            except Exception:
                return False
        while not is_roundtrip_valid(mol):
            try:
                # print("1)")
                Chem.SanitizeMol(mol)
                pass
            except Chem.KekulizeException as e:
                # print('2.1) KekulizeException:', e)
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Chem.AtomKekulizeException as e:
                # print('2.2) AtomKekulizeException:', e)
                # mol = Chem.MolFromSmiles(smiles, sanitize=False)
                for atom in mol.GetAtoms():
                    if not atom.IsInRing() and atom.GetIsAromatic():
                        atom.SetIsAromatic(False)
                for bond in mol.GetBonds():
                    if not bond.IsInRing() and (bond.GetIsAromatic() or bond.GetBondType() == Chem.rdchem.BondType.AROMATIC):
                        bond.SetIsAromatic(False)
                        bond.SetBondType(Chem.rdchem.BondType.SINGLE)
            except Chem.AtomValenceException as e:
                break
            except Exception as e:
                # print('3) Exception:', e)
                pass
            try:
                mol.UpdatePropertyCache()
            except Exception as e:
                # print('4) UpdatePropertyCache Exception:', e)
                pass
            count_while += 1
            if count_while > 10:
                break
        try:
            return Chem.MolToSmiles(mol)
        except Exception as e:
            # print('5) MolToSmiles Exception:', e)
            return None
    else:
        return None
    

def random_fragment_with_bond_atoms(smiles):
    if "." in smiles:
        frags = smiles.split(".")
        updated_frags = []
        for f in frags:
            try:
                if Chem.MolFromSmiles(f) is None or Chem.MolFromSmiles(f).GetNumBonds() == 0:
                    continue
                updated_frags.append(Chem.CanonSmiles(f))
            except:
                continue
        return updated_frags
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumBonds() == 0:
        return None

    # Pick a random non-ring bond
    non_ring_bonds = [b.GetIdx() for b in mol.GetBonds() if not b.IsInRing() and b.GetBondType() != Chem.rdchem.BondType.DATIVE]
    if not non_ring_bonds:
        # print("No non-ring bonds found.")
        return None
    bond_idx = random.choice(non_ring_bonds)
    bond = mol.GetBondWithIdx(bond_idx)
    a1_idx, a2_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    a1_symbol = mol.GetAtomWithIdx(a1_idx).GetSymbol()
    a2_symbol = mol.GetAtomWithIdx(a2_idx).GetSymbol()
    # print(f"Fragmenting non-ring bond between {a1_symbol} and {a2_symbol} at index {bond_idx}")

    # Break the bond
    fragged = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=True)
    
    # Get SMILES fragments
    frags = Chem.GetMolFrags(fragged, asMols=True, sanitizeFrags=True)

    if len(frags) < 2:
        # print("Fragmentation did not produce two fragments.")
        return None

    updated_frags = []
    # Convert to RWMol to allow atom removal
    frags_0_rw = Chem.RWMol(frags[0])
    frags_1_rw = Chem.RWMol(frags[1])

    # Remove dummy atoms ('*') from each fragment
    for atom in reversed(frags_0_rw.GetAtoms()):
        if atom.GetSymbol() == '*':
            atom_to_remove = atom.GetIdx()
    frags_0_rw.RemoveAtom(atom_to_remove)
    for atom in reversed(frags_1_rw.GetAtoms()):
        if atom.GetSymbol() == '*':
            atom_to_remove = atom.GetIdx()
    frags_1_rw.RemoveAtom(atom_to_remove)

    # Convert fragments to SMILES
    try:
        if frags_0_rw.GetNumBonds() > 0:
            frag_smiles = Chem.MolToSmiles(frags_0_rw, canonical=True)
            if Chem.MolFromSmiles(frag_smiles) is not None:
                updated_frags.append(frag_smiles)
    except:
        pass
    try:
        if frags_1_rw.GetNumBonds() > 0:
            frag_smiles = Chem.MolToSmiles(frags_1_rw, canonical=True)
            if Chem.MolFromSmiles(frag_smiles) is not None:
                updated_frags.append(frag_smiles)
    except:
        pass

    return updated_frags


####### PREPROCESSING HELPERS ##########

def tokenizer(smile):
    "Tokenizes SMILES string"
    # pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    pattern =  r"(\\+|\[[^\]]+]|->?|<-?|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|;|:|~|@|\?|>|\*|\|\$?|\$\|?|\$|\||\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    pattern_bracket = r"(\[(?:\d+)?[A-Za-z][a-z]?)([^\]]*\])"
    regezz_bracket = re.compile(pattern_bracket)
    tokens = []
    if smile is None or str(smile)=='nan':
        smile = ''
    try:
        for token in regezz.findall(smile):
            if "[" in token[0] and "]" in token[-1]:
                token_bracket = regezz_bracket.findall(token)
                if len(token_bracket) == 0:
                    tokens.append(token)
                else:
                    tokens.append(token_bracket[0][0])
                    tokens.append(token_bracket[0][1])
            else:
                tokens.append(token)
    except:
        print("ERROR: {} could not be tokenized".format(smile))
        tokens = []
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens

def build_org_dict(char_dict):
    org_dict = {}
    for i, (k, v) in enumerate(char_dict.items()):
        if i == 0:
            pass
        else:
            org_dict[int(v-1)] = k
    return org_dict

def encode_smiles(smile, max_len, char_dict):
    "Converts tokenized SMILES string to list of token ids"
    for i in range(max_len - len(smile)):
        if i == 0:
            smile.append('<end>')
        else:
            smile.append('_')
    smile_vec = []
    for c in smile:
        if c not in char_dict:
            print("WARNING: {} not in char_dict, replacing with <unk>".format(c))
            c = '<unk>'
        smile_vec.append(char_dict[c])
    return smile_vec

def get_char_weights(train_smiles, params, freq_penalty=0.5):
    "Calculates token weights for a set of input data"
    char_dist = {}
    char_counts = np.zeros((params['NUM_CHAR'],))
    char_weights = np.zeros((params['NUM_CHAR'],))
    for k in params['CHAR_DICT'].keys():
        char_dist[k] = 1 # 0
    for smile in train_smiles:
        for i, char in enumerate(smile):
            char_dist[char] += 1
        for j in range(i, params['MAX_LENGTH']):
            char_dist['_'] += 1
    for i, v in enumerate(char_dist.values()):
        char_counts[i] = v
    top = np.sum(np.log(char_counts))
    for i in range(char_counts.shape[0]):
        log_char_count = np.log(char_counts[i]) if char_counts[i] > 1 else 1
        char_weights[i] = top / log_char_count
    min_weight = char_weights.min()
    for i, w in enumerate(char_weights):
        if w > 2*min_weight:
            char_weights[i] = 2*min_weight
    scaler = MinMaxScaler((freq_penalty,1.0))
    char_weights = scaler.fit_transform(char_weights.reshape(-1, 1))
    return char_weights[:,0]


####### POSTPROCESSING HELPERS ##########

def decode_mols(encoded_tensors, org_dict):
    "Decodes tensor containing token ids into string"
    mols = []
    for i in range(encoded_tensors.shape[0]):
        encoded_tensor = encoded_tensors.cpu().numpy()[i,:] - 1
        mol_string = ''
        for i in range(encoded_tensor.shape[0]):
            idx = encoded_tensor[i]
            if org_dict[idx] == '<end>':
                break
            elif org_dict[idx] == '_':
                break
            else:
                mol_string += org_dict[idx]
        mol_string = mol_string.strip('_').strip('<end>').strip('.')
        mols.append(mol_string)
    return mols

def calc_reconstruction_accuracies(input_smiles, output_smiles):
    "Calculates SMILE, token and positional accuracies for a set of\
    input and reconstructed SMILES strings"
    max_len = 300 #254 #126
    smile_accs = []
    hits = 0
    misses = 0
    position_accs = np.zeros((2, max_len))
    for in_smi, out_smi in zip(input_smiles, output_smiles):
        if in_smi == out_smi:
            smile_accs.append(1)
        else:
            smile_accs.append(0)

        misses += abs(len(in_smi) - len(out_smi))
        for j, (token_in, token_out) in enumerate(zip(in_smi, out_smi)):
            if token_in == token_out:
                hits += 1
                position_accs[0,j] += 1
            else:
                misses += 1
            position_accs[1,j] += 1

    smile_acc = np.mean(smile_accs)
    token_acc = hits / (hits + misses)
    position_acc = []
    for i in range(max_len):
        position_acc.append(position_accs[0,i] / position_accs[1,i])
    return smile_acc, token_acc, position_acc

def calc_entropy(sample):
    "Calculates Shannon information entropy for a set of input memories"
    es = []
    for i in range(sample.shape[1]):
        probs, bin_edges = np.histogram(sample[:,i], bins=1000, range=(-5., 5.), density=True)
        es.append(entropy(probs))
    return np.array(es)

####### ADDITIONAL METRIC CALCULATIONS #########

def load_gen(path):
    "Loads set of generated SMILES strings from path"
    smiles = pd.read_csv(path).SMILES.to_list()
    return smiles

def valid(smiles, return_invalid=False):
    "Returns valid SMILES (RDKit sanitizable) from a set of SMILES strings"
    valid_smiles = []
    invalid_smiles = []
    for smi in smiles:
        current_smi = smi
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            smi = clean_kekulize(smi)
            if smi is None:
                invalid_smiles.append(current_smi)
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                invalid_smiles.append(current_smi)
                continue
        try:
            smi = normalize_smiles(smi)
            if smi is not None:
                valid_smiles.append(smi)
            else:
                invalid_smiles.append(current_smi)
        except Exception:
            invalid_smiles.append(current_smi)
                
    if return_invalid:
        return valid_smiles, invalid_smiles
    return valid_smiles

def calc_token_lengths(smiles):
    "Calculates the token lengths of a set of SMILES strings"
    lens = []
    for smi in smiles:
        smi = tokenizer(smi)
        lens.append(len(smi))
    return lens

def calc_MW(smiles):
    "Calculates the molecular weights of a set of SMILES strings"
    MWs = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        MWs.append(Descriptors.MolWt(mol))
    return MWs

def novel(smiles, train_smiles):
    "Returns novel SMILES strings that do not appear in training set"
    set_smiles = set(smiles)
    set_train = set(train_smiles)
    novel_smiles = list(set_smiles - set_train)
    return novel_smiles

def unique(smiles):
    "Returns unique SMILES strings from set"
    unique_smiles = set(smiles)
    return list(unique_smiles)

def fingerprints(smiles):
    "Calculates fingerprints of a list of SMILES strings"
    fps = np.zeros((len(smiles), 2048))
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        fp = np.asarray(fpgen.GetFingerprint(mol), dtype='uint8')
        fps[i,:] = fp
    return fps

def tanimoto_similarity(bv1, bv2):
    "Calculates Tanimoto similarity between two fingerprint bit vectors"
    mand = sum(bv1 & bv2)
    mor = sum(bv1 | bv2)
    return mand / mor

# def pass_through_filters(smiles, data_dir='data'):
#     """Filters SMILES strings based on method implemented in
#     http://nlp.seas.harvard.edu/2018/04/03/attention.html"""
#     _mcf = pd.read_csv('{}/mcf.csv'.format(data_dir))
#     _pains = pd.read_csv('{}/wehi_pains.csv'.format(data_dir), names=['smarts', 'names'])
#     _filters = [Chem.MolFromSmarts(x) for x in
#                 _mcf.append(_pains, sort=True)['smarts'].values]
#     filtered_smiles = []
#     for smi in smiles:
#         mol = Chem.MolFromSmiles(smi)
#         h_mol = Chem.AddHs(mol)
#         filtered = False
#         if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
#             filtered = True
#         if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
#             filtered = True
#         if not filtered:
#             filtered_smiles.append(smi)
#     return filtered_smiles

def cross_diversity(set1, set2, bs1=5000, bs2=5000, p=1, agg='max',
                    device='cpu'):
    """
    Function for calculating the maximum average tanimoto similarity score
    between the generated set and the training set (this code is adapted from
    https://github.com/molecularsets/moses)
    """
    agg_tanimoto = np.zeros(len(set2))
    total = np.zeros(len(set2))
    set2 = torch.tensor(set2).to(device).float()
    for j in range(0, set1.shape[0], bs1):
        x_stock = torch.tensor(set1[j:j+bs1]).to(device).float()
        for i in range(0, set2.shape[0], bs2):
            y_gen = set2[i:i+bs2]
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                   y_gen.sum(0, keepdim=True) -tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p!= 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i+y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i+y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i+y_gen.shape[1]] += jac.sum(0)
                total[i:i+y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return 1 - np.mean(agg_tanimoto)


def get_fingerprint_dictionary(smiles_list):
    result = {}
    for smiles in tqdm(smiles_list, desc="Get FPs"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = fpgen.GetFingerprint(mol)
            result[smiles] = fp
    return result


def similarity(a, b, radius=2, dictionary=None):
    if a is None or b is None: 
        return 0.0
    if dictionary and a in dictionary and b in dictionary:
        fp1 = dictionary[a]
        fp2 = dictionary[b]
    else:
        amol = Chem.MolFromSmiles(a)
        bmol = Chem.MolFromSmiles(b)
        if amol is None or bmol is None:
            # print(a, b)
            return 0.0
        fp1 = fpgen.GetFingerprint(amol)
        fp2 = fpgen.GetFingerprint(bmol)
    return DataStructs.TanimotoSimilarity(fp1, fp2) 


def internal_diversity(smiles_list, radius=2, dictionary=None):
    diversity_list = []
    for i, a in tqdm(enumerate(smiles_list), desc="IntDiv"):
        for b in smiles_list[i+1:]:
            diversity_list.append(1 - similarity(a, b, radius=radius, dictionary=dictionary))
    return np.mean(diversity_list), np.std(diversity_list)


def similarity_to_nearest_neighbor(smiles_list, ref_list, radius=2, dictionary=None):
    similarity_list = []
    for i, a in tqdm(enumerate(smiles_list), desc="SNN"):
        max_similarity = 0
        for b in ref_list:
            sim = similarity(a, b, radius=radius, dictionary=dictionary)
            if sim > max_similarity:
                max_similarity = sim
        similarity_list.append(max_similarity)
    return np.mean(similarity_list), np.std(similarity_list)


####### GRADIENT TROUBLESHOOTING #########

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    layers = np.array(layers)
    ave_grads = np.array(ave_grads)
    fig = plt.figure(figsize=(12,6))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.ylim(ymin=0, ymax=5e-3)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    return plt


####### MOLECULE VISUALIZATION #########

def display_molecule(molecules, col=5, title=None, texts=None):
    fig, axs = plt.subplots(math.ceil(len(molecules)/col), col, figsize=(3*col, math.ceil(len(molecules)*0.75)), dpi=300)
    fig.subplots_adjust(hspace=0.45, wspace=.0001, left=0, bottom=0.01, right=0.995, top=1)
    axs = axs.ravel()
    for i in range(math.ceil(len(molecules)/col)*col):
        if i < len(molecules):
            mol = molecules[i]
            ax = axs[i]
            ax.imshow(Chem.Draw.MolToImage(mol))
            ax.axis('off')
            if title:
                ax.set_title(title[i])
            if texts:
                ax.text(120, 350, texts[i], fontsize=10, ha='center')
        else:
            ax = axs[i]
            ax.axis('off')


####### MOLECULE HELPERS TECHNICAL ##########

# molecular weight
def get_mol_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)

# number of atoms
def get_num_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms()

# SA Score
def get_sa_score(smiles):
    import sys
    import os
    sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))
    from SA_Score import sascorer
    mol = Chem.MolFromSmiles(smiles)
    try:
        sascore = sascorer.calculateScore(mol)
    except Exception:
        print("Error calculating SA score for {}".format(smiles))
        return np.nan
    return sascore