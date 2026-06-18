import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import random
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from transvae.training_mol import TransVAE
from transvae.parsers import device_init, model_init, train_parser_mol
from transvae.tvae_util import build_org_dict, cleaner
from transvae.sampling import sampling
import multiprocessing as mp
from functools import partial

def train(args):
    print("Training with args:", args)
    device = device_init(args)
    
    ### Update beta init parameter
    start_epoch = 0
    args.checkpoint = None if args.checkpoint == str("none") else args.checkpoint
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if args.finetune.lower() == 'true':
            print("Finetuning mode: not updating beta_init based on checkpoint epoch.")
        else:
            start_epoch = ckpt['epoch']+1
    
    if os.path.exists(os.path.join(args.data_dir, args.data_source, f"split_{args.seed}.csv")):
        print("Loading existing train/val/test split from CSV...")
        df_all = pd.read_csv(os.path.join(args.data_dir, args.data_source, f"split_{args.seed}.csv"))
    else:
        ### Load data, vocab and token weights
        if args.data_source == 'pubchem' or args.data_source == 'pubchem10M':
            import gzip
            from itertools import islice
            # with gzip.open(os.path.join(args.data_dir, args.data_source, "CID-SMILES.gz"), "rt", encoding="utf-8") as f:
            #     content = f.read()
            #     mols = [line.split('\t')[1] for line in content.splitlines() if line.strip()] # assuming the file has two columns: CID and SMILES
            # mols = mols[:1000000] # use only first 1M molecules for training and vocab building
            mols = []
            file_path = os.path.join(args.data_dir, args.data_source, "CID-SMILES.gz")
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                data_size = 11000000 if args.data_source == 'pubchem10M' else 5000000
                for line in tqdm(islice(f, 0, data_size), desc="Reading dataset"):
                    if line:
                        parts = line.split('\t', 1)
                        if len(parts) > 1:
                            mols.append(parts[1].rstrip('\n'))
        else:
            df = pd.read_csv(os.path.join(args.data_dir, args.data_source, args.data_source+'.smi'), header=None, names=['smiles'])
            mols = df['smiles'].tolist()
            
        # train, val, test split
        # train 70%, val 10%, test 20%
        dataset_size = len(mols)
        print("Original dataset size:", dataset_size)
        # clean and filter molecules
        # 1. remove invalid SMILES
        # 2. remove molecules with more than 300 tokens (after encoding)
        # 3. remove isotopes
        # mols = [cleaned for smi in tqdm(mols, desc="Cleaning molecules") if (cleaned := cleaner(smi, max_len=args.max_len)) is not None]
        # use multiprocessing to speed up cleaning
        try:
            ctx = mp.get_context('fork')
        except Exception:
            ctx = mp.get_context('spawn')
        pool_size = max(1, mp.cpu_count() - 1)
        with ctx.Pool(processes=pool_size) as pool:
            func = partial(cleaner, max_len=args.max_len)
            results = list(tqdm(pool.imap(func, mols, chunksize=256), total=len(mols), desc="Cleaning molecules"))
        mols = [r for r in results if r is not None]

        dataset_size_cleaned = len(mols)
        print("After cleaning, dataset size:", dataset_size_cleaned)
        df = pd.DataFrame({'smiles': mols})
        df['id'] = df.index
        df_trainval, df_test = train_test_split(df, test_size=0.05, random_state=args.seed)
        dataset_size_trainval = len(df_trainval)
        val_size = 0.05 * dataset_size_cleaned / dataset_size_trainval
        df_train, df_val = train_test_split(df_trainval, test_size=val_size, random_state=args.seed)
        df_train['s'] = 'train'
        df_val['s'] = 'val'
        df_test['s'] = 'test'
        df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
        df_all.to_csv(os.path.join(args.data_dir, args.data_source, f"split_{args.seed}.csv"), index=False)


    df_train = df_all[df_all['s']=='train']
    # df_train_original = df_train.copy() 
    df_val = df_all[df_all['s']=='val']
    df_test = df_all[df_all['s']=='test']
    train_idx = df_train['id'].to_numpy()
    val_idx = df_val['id'].to_numpy()
    test_idx = df_test['id'].to_numpy()
    train_mols = df_train['smiles'].to_numpy()
    val_mols = df_val['smiles'].to_numpy()
    test_mols = df_test['smiles'].to_numpy()

    with open(args.vocab_path, 'rb') as f:
        char_dict = pickle.load(f)
    if args.char_weights_path is not None:
        char_weights = np.load(args.char_weights_path)
        args.char_weights = char_weights

    org_dict = build_org_dict(char_dict)
    args.char_dict = char_dict
    args.org_dict = org_dict

    print("[AFTER] Train size: {}, Val size: {}, Test size: {}".format(len(train_mols), len(val_mols), len(test_mols)))

    ### Train model
    if args.checkpoint is None:
        vae = model_init(args, mode="training")
    else:
        vae = model_init(args, mode="training", load_fn=args.checkpoint)
    vae.train(train_idx, train_mols, val_idx, val_mols, epochs=args.epochs-start_epoch, save_freq=args.save_freq)

    ### Test model
    ckpt_path = os.path.join(args.data_dir, args.data_source, 'checkpoints', 'best_'+vae.name+'.ckpt')
    vae.load(checkpoint_path=ckpt_path, mode="inference")
    print("Loaded best model from:", ckpt_path)
    print("Best val loss:", vae.best_loss)
    print("Best epoch:", vae.best_epoch)
    vae.model.eval()


if __name__ == '__main__':
    parser = train_parser_mol()
    args = parser.parse_args()
    train(args)
