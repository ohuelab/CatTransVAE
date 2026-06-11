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
from transvae.tvae_util import build_org_dict, cleaner, random_fragment_with_bond_atoms
from transvae.sampling import sampling

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
            # total_epochs = args.epochs 
            # beta_init = (args.beta - args.beta_init) / total_epochs * start_epoch
            # args.beta_init = beta_init
    
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
                for line in tqdm(islice(f, 0, 5000000), desc="Reading dataset"):
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
        mols = [cleaned for smi in tqdm(mols, desc="Cleaning molecules") if (cleaned := cleaner(smi, max_len=args.max_len)) is not None]
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
    df_train_original = df_train.copy() 
    df_val = df_all[df_all['s']=='val']
    df_test = df_all[df_all['s']=='test']
    train_idx = df_train['id'].to_numpy()
    val_idx = df_val['id'].to_numpy()
    test_idx = df_test['id'].to_numpy()
    train_mols = df_train['smiles'].to_numpy()
    val_mols = df_val['smiles'].to_numpy()
    test_mols = df_test['smiles'].to_numpy()
    
    print("[BEFORE] Train size: {}, Val size: {}, Test size: {}".format(len(train_mols), len(val_mols), len(test_mols)))

    # Expansion
    if args.expansion != "none":
        if os.path.exists('data/{}/expansion_{}.csv'.format(args.data_source, args.seed)):
            df_similar = pd.read_csv('data/{}/expansion_{}.csv'.format(args.data_source, args.seed))
        else:
            if args.expansion == 'pubchem' or args.expansion == 'pubchem10M':
                import gzip
                from itertools import islice
                # with gzip.open(os.path.join(args.data_dir, args.expansion, "CID-SMILES.gz"), "rt", encoding="utf-8") as f:
                #     content = f.read()
                #     mols = [line.split('\t')[1] for line in content.splitlines() if line.strip()] # assuming the file has two columns: CID and SMILES
                # mols = mols[1000000:2000000] # use only next 1M molecules for training and vocab building
                mols = []
                file_path = os.path.join(args.data_dir, args.expansion, "CID-SMILES.gz")
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    for line in tqdm(islice(f, 1000000, 2000000), desc="Reading expansion dataset"):
                        if line:
                            parts = line.split('\t', 1)
                            if len(parts) > 1:
                                mols.append(parts[1].rstrip('\n'))
            else:
                df = pd.read_csv(os.path.join(args.data_dir, args.expansion, args.expansion+'.csv'))
                mols = df['smiles'].tolist()
            
            # clean and filter molecules
            # 1. remove invalid SMILES
            # 2. remove molecules with more than 300 tokens (after encoding)
            # 3. remove isotopes
            mols = [cleaned for smi in tqdm(mols, desc="Cleaning expansion molecules") if (cleaned := cleaner(smi, max_len=args.max_len)) is not None]
            df_pretrain = pd.DataFrame({'smiles': mols})

            similar_mols = []
            similar_idx = []
            similar_limit = int(len(df_train) * 0.2)

            # shuffle pretrain dataset
            # df_pretrain = df_pretrain.sample(frac=1).reset_index(drop=True)
            similar_mols = df_pretrain['smiles'].tolist()[:similar_limit]
            similar_idx = [f"{args.expansion}_{i}" for i in range(similar_limit)]


            print(f"Total similar molecules found: {len(similar_mols)}")
            df_similar = pd.DataFrame({'id': similar_idx, 'smiles': similar_mols, 's': 'train'})
            df_similar.to_csv('data/{}/expansion_{}.csv'.format(args.data_source, args.seed), index=False)
        
        print(f"Number of similar molecules found: {len(df_similar)}")
        df_train = pd.concat([df_train, df_similar], ignore_index=True)
        print(f"New training set size after expansion: {len(df_train)}")

    train_idx = df_train['id'].to_numpy()
    val_idx = df_val['id'].to_numpy()
    test_idx = df_test['id'].to_numpy()
    train_mols = df_train['smiles'].to_numpy()
    val_mols = df_val['smiles'].to_numpy()
    test_mols = df_test['smiles'].to_numpy()
    
    print("[AFTER EXPANSION] Train size: {}, Val size: {}, Test size: {}".format(len(train_mols), len(val_mols), len(test_mols)))

    # Augmentation
    if args.augmentation != 0:
        if os.path.exists('data/{}/augmentation_{}.csv'.format(args.data_source, args.seed)):
            df_augmented = pd.read_csv('data/{}/augmentation_{}.csv'.format(args.data_source, args.seed))
        else:
            augmented_mols = []
            augmented_idx = []
            for index, row in tqdm(df_train.iterrows(), desc="Augmenting training molecules"):
                smi = row['smiles']
                idx = row['id']
                if pd.isna(smi):
                    continue
                if Chem.MolFromSmiles(smi) is None:
                    continue

                dup_aug = [smi]
                for i in range(args.augmentation):
                    try:
                        aug_mol = Chem.MolToSmiles(Chem.MolFromSmiles(smi), doRandom=True)
                    except:
                        continue
                    if aug_mol is None:
                        continue
                    if aug_mol not in dup_aug: # avoid duplicates
                        augmented_mols.append(aug_mol)
                        augmented_idx.append(f"{idx}_aug{i+1}")
                    dup_aug.append(aug_mol)
            
            print(f"Total augmented molecules generated: {len(augmented_mols)}")
            df_augmented = pd.DataFrame({'id': augmented_idx, 'smiles': augmented_mols, 's': 'train'})
            df_augmented.to_csv('data/{}/augmentation_{}.csv'.format(args.data_source, args.seed), index=False)

        print(f"Number of augmented molecules generated: {len(df_augmented)}")
        df_train = pd.concat([df_train, df_augmented], ignore_index=True)
        print(f"New training set size after augmentation: {len(df_train)}")


    train_idx = df_train['id'].to_numpy()
    val_idx = df_val['id'].to_numpy()
    test_idx = df_test['id'].to_numpy()
    train_mols = df_train['smiles'].to_numpy()
    val_mols = df_val['smiles'].to_numpy()
    test_mols = df_test['smiles'].to_numpy()

    print("[AFTER AUGMENTATION] Train size: {}, Val size: {}, Test size: {}".format(len(train_mols), len(val_mols), len(test_mols)))

    ### Train model
    vae = model_init(args, mode="training", load_fn=args.checkpoint, finetune=args.finetune.lower() == 'true')
    vae.train(train_idx, train_mols, val_idx, val_mols, epochs=args.epochs-start_epoch, save_freq=args.save_freq)

    ### Test model
    ckpt_path = os.path.join(args.data_dir, args.data_source, 'checkpoints', 'best_'+vae.name+'.ckpt')
    vae.load(checkpoint_path=ckpt_path, mode="inference")
    print("Loaded best model from:", ckpt_path)
    print("Best val loss:", vae.best_loss)
    print("Best epoch:", vae.best_epoch)
    # vae.model.eval()
    # vae.test(test_idx, test_mols)

    ### Number of paramaters
    total_params = sum(p.numel() for p in vae.model.parameters())
    trainable_params = sum(p.numel() for p in vae.model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print("#############################")
    print()

if __name__ == '__main__':
    parser = train_parser_mol()
    args = parser.parse_args()
    train(args)