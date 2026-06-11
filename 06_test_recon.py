import os
import pickle
from unicodedata import normalize
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
import matplotlib.pyplot as plt
from transvae.sampling import reconstructing, sampling
from transvae.training_mol import TransVAE
from transvae.parsers import device_init, model_init, sample_parser_mol
from transvae.tvae_util import *

def sample(args):
    print("Training with args:", args)
    device = device_init(args)
    
    print("### Loading data and splits...")
    if os.path.exists(os.path.join(args.data_dir, args.data_source, f"split_{args.seed}.csv")):
        print("Loading existing train/val/test split from CSV...")
        test_file = os.path.join(args.data_dir, args.data_source, f"split_{args.seed}.csv")
        df_all = pd.read_csv(test_file)
        
        df_train = df_all[df_all['s']=='train']
        df_val = df_all[df_all['s']=='val']
        df_test = df_all[df_all['s']=='test']
        # train_idx = df_train['id'].to_numpy()
        # val_idx = df_val['id'].to_numpy()
        # test_idx = df_test['id'].to_numpy()
        train_mols = df_train['smiles'].to_numpy()
        val_mols = df_val['smiles'].to_numpy()
        test_mols = df_test['smiles'].to_numpy()

        print("[FINAL] Train size: {}, Val size: {}, Test size: {}".format(len(train_mols), len(val_mols), len(test_mols)))
        print("#############################")
        print()
    else:
        if os.path.exists(os.path.join(args.data_dir, args.data_source, f"{args.data_source}.csv")):
            test_file = os.path.join(args.data_dir, args.data_source, f"{args.data_source}.csv")
            print("Loading existing dataset from CSV...")
            df_test = pd.read_csv(os.path.join(test_file))
            test_mols = df_test['smiles'].to_numpy()
            print("[FINAL] Test size: {}".format(len(test_mols)))
        if os.path.exists(os.path.join(args.data_dir, args.data_source, f"{args.data_source}.smi")):
            test_file = os.path.join(args.data_dir, args.data_source, f"{args.data_source}.smi")
            print("Loading existing dataset from .smi...")
            test_mols = []
            with open(test_file, 'r') as f:
                test_mols = [line.strip() for line in f]
            print("[FINAL] Test size: {}".format(len(test_mols)))
        else:
            print("No existing dataset found. Please put dataset in .csv or .smi format or\
                   run 02_train_mol.py first to create train/val/test splits and save them as CSV files")
            return

    ### Test model
    print("### Loading model...")
    ckpt_path = args.checkpoint
    vae = TransVAE(args, mode="inference", load_fn=ckpt_path)
    print("Loaded best model from:", ckpt_path)
    print("Best val loss:", vae.best_loss)
    print("Best epoch:", vae.best_epoch)
    vae.model.eval()
    print("#############################")
    print()

    ### Make directory
    save_name = f"{args.seed}_{args.save_name}"
    os.makedirs(os.path.join('results', 
                             'reconstruction'), 
                             exist_ok=True)

    ### Test reconstruction
    print("### Testing reconstruction...")
    test_mols_subset = test_mols
    decode_method = args.decode_method
    save_path = os.path.join('results', 
                             'reconstruction',
                             f'{vae.name}_{save_name}_{decode_method}.csv')
    results, metrics_recon = reconstructing(vae, 
                            test_mols_subset, 
                            decode_method=decode_method, 
                            metrics=True, 
                            save_path=save_path)
    
    # save comparison metrics
    save_path = os.path.join('results', 
                             'reconstruction', 
                             f'_metrics_comparison.csv')
    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            f.write("setting,name,model,"
                    "checkpoint,version,dataset,test_file,test_size,decode_method,"
                    "recon_match,recon_match_percentage,recon_similarity,recon_similarity_std\n")
    with open(save_path, 'a') as f:
        f.write(f"{save_name},{vae.name},,"
                f"{ckpt_path},,{args.data_source},{test_file},{metrics_recon['Total'][0]},{decode_method},"
                f"{metrics_recon['Matching'][0]},{metrics_recon['Matching'][1]},{metrics_recon['Similarity'][0]},{metrics_recon['Similarity'][1]}\n")
    
    print("#############################")
    print()

if __name__ == '__main__':
    parser = sample_parser_mol()
    args = parser.parse_args()
    sample(args)
