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
from transvae.sampling import reconstructing, sampling, sampling_analysis
from transvae.training_mol import TransVAE
from transvae.parsers import device_init, model_init, sample_parser_mol
from transvae.tvae_util import *
from itertools import product

def sample(args):
    print("Training with args:", args)
    device = device_init(args)
    
    print("### Loading data and splits...")
    if os.path.exists(os.path.join(args.data_dir, args.data_source, f"split_{args.seed}.csv")):
        print("Loading existing train/val/test split from CSV...")
        df_all = pd.read_csv(os.path.join(args.data_dir, args.data_source, f"split_{args.seed}.csv"))
    else:
        print("No existing split found. Please run 02_train_mol.py first to create train/val/test splits and save them as CSV files.")
        return

    df_train = df_all[df_all['s']=='train']
    df_val = df_all[df_all['s']=='val']
    df_test = df_all[df_all['s']=='test']
    # train_idx = df_train['id'].to_numpy()
    # val_idx = df_val['id'].to_numpy()
    # test_idx = df_test['id'].to_numpy()
    train_mols = df_train['smiles'].to_numpy()
    val_mols = df_val['smiles'].to_numpy()
    test_mols = df_test['smiles'].to_numpy()

    print("[AFTER] Train size: {}, Val size: {}, Test size: {}".format(len(train_mols), len(val_mols), len(test_mols)))
    print("#############################")
    print()


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

    # decode_method_choices = ['greedy']
    # sample_mode_choices = ['rand', 'k_high_entropy', 'rand_training', 'rand_target']
    # # entropy_cutoff_choices = [1, 2, 5]
    # k_entropy_choices = [10, 30, 50, 100] # Only for k_high_entropy
    # temperature_choices = [0.3, 0.7, 1.0, 1.3, 2.0] # Only for do_sample=True
    # top_k_choices = [1, 3, 5, 10] # Only for do_sample=True
    # do_sample_choices = [True]
    # dummy_attaches_enabled_choices = [False]

    combination = [

        ('greedy', 'rand', 10, 1.0, -1, True, False),
        ('greedy', 'k_high_entropy', 10, 1.0, -1, True, False),
        ('greedy', 'rand_target', 10, 1.0, -1, True, False),
        ('beam', 'rand', 10, 1.0, -1, True, False),
        ('beam', 'k_high_entropy', 10, 1.0, -1, True, False),
        ('beam', 'rand_target', 10, 1.0, -1, True, False),

        ('greedy', 'rand', 10, 1.3, -1, True, False),
        ('greedy', 'rand', 10, 2.0, -1, True, False),
        ('greedy', 'rand', 10, 0.7, -1, True, False),
        ('greedy', 'rand', 10, 0.1, -1, True, False),
        ('beam', 'rand', 10, 1.3, -1, True, False),
        ('beam', 'rand', 10, 2.0, -1, True, False),
        ('beam', 'rand', 10, 0.7, -1, True, False),
        ('beam', 'rand', 10, 0.1, -1, True, False),

        ('greedy', 'k_high_entropy', 50, 1.0, -1, True, False),
        ('greedy', 'k_high_entropy', 100, 1.0, -1, True, False),
        ('greedy', 'k_high_entropy', 256, 1.0, -1, True, False),
        ('beam', 'k_high_entropy', 50, 1.0, -1, True, False),
        ('beam', 'k_high_entropy', 100, 1.0, -1, True, False),
        ('beam', 'k_high_entropy', 256, 1.0, -1, True, False),

        ('greedy', 'rand', 10, 1.0, -1, False, False),
        ('beam', 'rand', 10, 1.0, -1, False, False),

        ('greedy', 'rand', 10, 1.0, 5, True, False),
        ('greedy', 'rand', 10, 1.0, 10, True, False),
        ('greedy', 'rand', 10, 1.0, 25, True, False),
        ('beam', 'rand', 10, 1.0, 5, True, False),
        ('beam', 'rand', 10, 1.0, 10, True, False),
        ('beam', 'rand', 10, 1.0, 25, True, False),

        ('greedy', 'rand_training', 10, 1.0, -1, True, False),
        ('beam', 'rand_training', 10, 1.0, -1, True, False),
    ]
    
    for c in tqdm(combination, desc="Testing combinations of sampling settings"):
        decode_method_selected, sample_mode_selected, k_entropy_selected, temperature_selected, top_k_selected, do_sample_selected, dummy_attaches_enabled_selected = c
        print(f"### Testing combination: \
                decode_method={decode_method_selected}, \
                sample_mode={sample_mode_selected}, \
                k_entropy={k_entropy_selected}, \
                temperature={temperature_selected}, \
                top_k={top_k_selected}, \
                do_sample={do_sample_selected}, \
                dummy_attaches_enabled={dummy_attaches_enabled_selected}...")


        ### Make directory
        save_name = f"{args.seed}_{args.save_name}"
        os.makedirs(os.path.join(args.data_dir, 
                                args.data_source, 
                                'generated', 
                                f'{vae.name}_{save_name}_test'), 
                                exist_ok=True)

        ### Sampling after training
        print("### Testing sampling...")
        train_mols_sample = train_mols

        sample_mode = sample_mode_selected
        decode_method = decode_method_selected
        entropy_cutoff = args.entropy_cutoff
        k_entropy = k_entropy_selected
        temperature = temperature_selected
        top_k = top_k_selected
        do_sample = do_sample_selected
        n_samples = args.n_samples
        n_samples_per_batch = args.n_samples_per_batch
        # total samples = n_samples * n_samples_per_batch
        dummy_attaches_enabled = dummy_attaches_enabled_selected
        prompt = 'none'
        if 'none' not in str(args.prompt):
            prompt = args.prompt.split(',')

        save_path = os.path.join(args.data_dir, 
                                args.data_source, 
                                'generated', 
                                f'{vae.name}_{save_name}_test',
                                f'{str(c)}.csv')
        samples, metrics_sampling = sampling(vae, 
                                    sample_mode=sample_mode,
                                    decode_method=decode_method,
                                    n_samples=n_samples, 
                                    n_samples_per_batch=n_samples_per_batch,
                                    prompt=prompt, 
                                    k_entropy=k_entropy, 
                                    entropy_cutoff=entropy_cutoff,
                                    temperature=temperature,
                                    top_k=top_k,
                                    do_sample=do_sample,
                                    dummy_attaches_enabled=dummy_attaches_enabled,
                                    ref_mols=train_mols_sample,
                                    metrics=True, 
                                    save_path=save_path,
                                    seed=args.seed)
        metrics_analysis = sampling_analysis(samples, save_path, metrics=True)
        
        # save comparison metrics
        save_path = os.path.join(args.data_dir, 
                                args.data_source, 
                                'generated', 
                                f'{vae.name}_{save_name}_test',
                                f'_metrics_comparison.csv')
        if not os.path.exists(save_path):
            with open(save_path, 'w') as f:
                f.write("model,setting,prompt,sample_mode,decode_method,"
                        "k_entropy,entropy_cutoff,temperature,top_k,do_sample,n_samples,n_samples_per_batch,"
                        # "recon_match,recon_match_percentage,recon_similarity,recon_similarity_std,"
                        "valid,valid_percentage,"
                        "unique_total,unique_total_percentage,unique_valid,unique_valid_percentage,"
                        "novel_total,novel_total_percentage,novel_validunique,novel_validunique_percentage,"
                        "intdiv,intdiv_std,snn,snn_std,"
                        "mol_weight_avg,mol_weight_std,num_atoms_avg,num_atoms_std,sascore_avg,sascore_std\n")
        with open(save_path, 'a') as f:
            f.write(f"{vae.name},{save_name},{prompt},{sample_mode},{decode_method},"
                    f"{k_entropy},{entropy_cutoff},{temperature},{top_k},{do_sample},{n_samples},{n_samples_per_batch},"
                    # f"{metrics_recon['Matching'][0]},{metrics_recon['Matching'][1]},{metrics_recon['Similarity'][0]},{metrics_recon['Similarity'][1]},"
                    f"{metrics_sampling['Valid'][0]},{metrics_sampling['Valid'][1]},"
                    f"{metrics_sampling['Unique_Total'][0]},{metrics_sampling['Unique_Total'][1]},{metrics_sampling['Unique_Valid'][0]},{metrics_sampling['Unique_Valid'][1]},"
                    f"{metrics_sampling['Novel_Total'][0]},{metrics_sampling['Novel_Total'][1]},{metrics_sampling['Novel_ValidUnique'][0]},{metrics_sampling['Novel_ValidUnique'][1]},"
                    f"{metrics_sampling['IntDiv'][0]},{metrics_sampling['IntDiv'][1]},{metrics_sampling['SNN'][0]},{metrics_sampling['SNN'][1]},"
                    f"{metrics_analysis['MW'][0]},{metrics_analysis['MW'][1]},{metrics_analysis['NumAtoms'][0]},{metrics_analysis['NumAtoms'][1]},{metrics_analysis['SAScore'][0]},{metrics_analysis['SAScore'][1]}\n")
        
    print("#############################")
    print()


if __name__ == '__main__':
    parser = sample_parser_mol()
    args = parser.parse_args()
    sample(args)
