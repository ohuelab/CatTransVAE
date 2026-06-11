import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from transvae.tvae_util import *
from transvae.parsers import vocab_parser
from data._dataset import dataset_args
from multiprocessing import Pool
from rdkit import Chem
import pandas as pd

dictbase = []
with open('data/_dictbase.txt', 'r') as f:
    dictbase = [line.strip().split('\t')[0] for line in f if line.strip()]

def process_mol(args_tuple):
    line, max_len = args_tuple
    if pd.isna(line) or str(line).strip() == '' or str(line).strip().lower() == 'nan':
        return None
    line = str(line).strip().split('\n')[0]
    if len(line) > max_len * 2:
        return None
    smiles = normalize_smiles(line)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
        smiles = Chem.MolToSmiles(mol)
    except Exception:
        return None
    token = tokenizer(smiles)
    if len(token) > max_len:
        return None
    return token


def build_vocab_mol(args):
    # build train / test file
    print('loading data...')
    if args.data_source == 'pubchem10M':
        import gzip
        with gzip.open(os.path.join(args.data_dir, args.data_source, "CID-SMILES.gz"), "rt", encoding="utf-8") as f:
            content = f.read()
            mols = [line.split('\t')[1] for line in content.splitlines() if line.strip()] # assuming the file has two columns: CID and SMILES
    else:
        df = pd.read_csv(os.path.join(args.data_dir, args.data_source, args.data_source+'.smi'), header=None, names=['smiles'])
        mols = df['smiles'].tolist()

    ### Build vocab dictionary
    print('building dictionary...')
    print(f'num molecules: {len(mols)}')

    char_dict = {'<start>': 0}
    char_set = set()
    char_idx = 1
    mol_toks = []
    max_len = args.max_len
    limit = 100000000
    # limit = 10000
    mols = mols[:limit] if limit < len(mols) else mols
    max_len_seen = 0

    with Pool() as pool:
        print(f'Using {pool._processes} processes to process {len(mols)} molecules...')
        args_iter = ((line, max_len) for line in mols)
        results = pool.imap_unordered(process_mol, args_iter, chunksize=100000)
        for token in tqdm(results, total=len(mols)):
            if token is not None:
                mol_toks.append(token)
                max_len_seen = max(max_len_seen, len(token))
                char_set.update(token)   # SAFE: main process only

    # Additional dict base
    for tok in dictbase:
        if tok not in char_set:
            char_set.add(tok)
    # sort dictionary key characters and re-index
    sorted_keys = sorted(char_set)
    for k in sorted_keys:
        char_dict[k] = char_idx
        char_idx += 1
    # add padding and end token
    char_dict['_'] = char_idx
    char_dict['<unk>'] = char_idx + 1
    char_dict['<end>'] = char_idx + 2

    ### write dictionary in text file
    with open(os.path.join(args.data_dir, args.vocab_name+'.txt'), 'w') as f:
        for k, v in char_dict.items():
            f.write(f'{k}\t{v}\n')

    ### Write dictionary to file
    with open(os.path.join(args.data_dir, args.vocab_name+'.pkl'), 'wb') as f:
        pickle.dump(char_dict, f)

    # keep metadata
    with open(os.path.join(args.data_dir, args.vocab_name+'_metadata.txt'), 'w') as f:
        f.write(f'NUM_CHAR\t{len(char_dict.keys())}\n')
        f.write(f'MAX_LENGTH\t{max_len_seen}\n')

    ### Set weights params
    del char_dict['<start>']
    # round up to next power of two (e.g. 230 -> 256)
    next_pow2 = 2**math.ceil(math.log2(max_len_seen))
    params = {'MAX_LENGTH': next_pow2,
              'NUM_CHAR': len(char_dict.keys()),
              'CHAR_DICT': char_dict}
    print(params)

    ### Calculate weights
    print('calculating weights...')
    char_weights = get_char_weights(mol_toks, params, freq_penalty=args.freq_penalty)
    char_weights[-3] = args.pad_penalty # pad token

    # save weights to file
    np.save(os.path.join(args.data_dir, args.weights_name+'.npy'), char_weights)

    ### Write weights to file
    with open(os.path.join(args.data_dir, args.weights_name+'.txt'), 'w') as f:
        for c, w in zip(char_dict.keys(), char_weights):
            f.write(f'{c}\t{w}\n')

    # check if pubchem exists
    if args.data_source != 'pubchem10M' and os.path.exists(os.path.join(args.data_dir, "pubchem10M", "pubchem10M_dict.pkl")):
        with open(os.path.join(args.data_dir, "pubchem10M", "pubchem10M_dict.pkl"), 'rb') as f:
            char_dict = pickle.load(f)
        max_len_seen = int(open(os.path.join(args.data_dir, "pubchem10M", "pubchem10M_dict_metadata.txt"), 'r').read().splitlines()[1].split('\t')[1])

        with open(os.path.join(args.data_dir, args.vocab_name+'_pubchem.pkl'), 'wb') as f:
            pickle.dump(char_dict, f)

        # keep metadata
        with open(os.path.join(args.data_dir, args.vocab_name+'_metadata_pubchem.txt'), 'w') as f:
            f.write(f'NUM_CHAR\t{len(char_dict.keys())}\n')
            f.write(f'MAX_LENGTH\t{max_len_seen}\n')
    
        ### Set weights params
        del char_dict['<start>']
        # round up to next power of two (e.g. 230 -> 256)
        next_pow2 = 2**math.ceil(math.log2(max_len_seen))
        params = {'MAX_LENGTH': next_pow2,
                'NUM_CHAR': len(char_dict.keys()),
                'CHAR_DICT': char_dict}
        print(params)

        ### Calculate weights
        print('calculating weights...')
        char_weights = get_char_weights(mol_toks, params, freq_penalty=args.freq_penalty)
        char_weights[-3] = args.pad_penalty # pad token
        
        # average weights with pubchem weights for shared characters
        char_weights_pubchem = np.load(os.path.join(args.data_dir, "pubchem", "pubchem_weight.npy"))
        # average weights with pubchem weights for shared characters
        print(len(char_weights), len(char_weights_pubchem))
        assert len(char_weights) == len(char_weights_pubchem)
        char_weights = (char_weights + char_weights_pubchem) / 2

        # save weights to file
        np.save(os.path.join(args.data_dir, args.weights_name+'_pubchem.npy'), char_weights)

        ### Write weights to file
        with open(os.path.join(args.data_dir, args.weights_name+'_pubchem.txt'), 'w') as f:
            for c, w in zip(char_dict.keys(), char_weights):
                f.write(f'{c}\t{w}\n')



if __name__ == '__main__':
    parser = vocab_parser()
    args = parser.parse_args()
    if args.data_type == 'mol':
        build_vocab_mol(args)
    # elif args.data_type == 'rxn':
    #     build_vocab_rxn(args)
    else:
        raise ValueError(f'Invalid data type: {args.data_type}')
