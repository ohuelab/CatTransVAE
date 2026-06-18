from transvae.tvae_util import *
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from rdkit import Chem
from xpromptsmiles.xprompt import *
from functools import partial
from time import perf_counter

def reconstructing(vae, 
                   test_mols, 
                   decode_method="greedy", 
                   metrics=True, 
                   save_path=None):
    
    test_idx = list(range(len(test_mols)))
    recon_mols = vae.reconstruct(test_idx, test_mols, method=decode_method)
    print("Reconstructing with TransVAE...", len(recon_mols), recon_mols[:10])
    recon_df = pd.DataFrame({'original': test_mols, 'reconstructed': recon_mols})
    if metrics:
        matches = sum(1 for original, recon in zip(test_mols, recon_mols) if original == recon)
        matches_percentage = matches / len(test_mols) if len(test_mols) > 0 else 0
        print(f"Matching rate: {matches}/{len(test_mols)} = {matches_percentage:.2%}")
        similar = []
        for original, recon in zip(test_mols, recon_mols):
            sim = similarity(original, recon)
            similar.append(sim)
        similarity_avg = np.mean(similar)
        similarity_std = np.std(similar)
        print(f"Average similarity: {similarity_avg:.4f}")
        if save_path is not None:
            with open(save_path.replace('.csv', '_metrics_recon.txt'), 'w') as f:
                f.write(f"Total\t{len(test_mols)}\t{1.0:.4%}\n")
                f.write(f"Matching\t{matches}\t{matches_percentage:.4%}\n")
                f.write(f"Average Similarity\t{similarity_avg:.4f}\t{similarity_std:.4f}\n")

    if save_path is not None:
        recon_df.to_csv(save_path, index=False)

        pairs_to_visualize = recon_df.sample(n=10) if len(recon_df) >= 10 else recon_df
        mols = []
        for _, row in pairs_to_visualize.iterrows():
            original_mol = Chem.MolFromSmiles(row['original'])
            if original_mol is None:
                original_mol = Chem.MolFromSmiles(row['original'], sanitize=False)
            recon_mol = Chem.MolFromSmiles(row['reconstructed'])
            if recon_mol is None:
                recon_mol = Chem.MolFromSmiles(row['reconstructed'], sanitize=False)
            if original_mol is not None and recon_mol is not None:
                mols.append((original_mol, recon_mol))
        if len(mols) > 0:
            pair_mols = []
            for pair in mols:
                pair_mols.extend([pair[0], pair[1]])
            # img = Draw.MolsToGridImage(pair_mols, molsPerRow=2, subImgSize=(300,300), legends=['Original', 'Reconstruction']*len(mols))
            display_molecule(pair_mols, col=2, texts=['Original', 'Reconstruction']*len(mols))
            plt.savefig(save_path.replace('.csv', '.png'))
        else:
            print("No valid molecules in reconstruction")
    
    if metrics:
        return recon_df, {"Total": (len(test_mols), 1),
                         "Matching": (matches, matches_percentage),
                         "Similarity": (similarity_avg, similarity_std)}
    else:
        return recon_df


def sampling(vae, 
             sample_mode, 
             decode_method="greedy", 
             sample_condition=None, 
             n_samples=1, 
             n_samples_per_batch=100, 
             prompt='none', 
             k_entropy=0, 
             entropy_cutoff=2.5, 
             limit_max_len=300,
             temperature=1.0,
             top_k=None,
             do_sample=False,
             dummy_attaches_enabled=True,
             ref_mols=None, 
             metrics=True, 
             save_path=None,
             seed=0):
    
    if prompt is None or 'none' in str(prompt):
        prompt = None
    elif "*" in str(prompt) and "," not in str(prompt):
        prompt = prompt[0]
    elif 'none' not in str(prompt):
        prompt = prompt[0]
        # prompt = prompt.split(',')

    if sample_mode in ['high_entropy', 'k_high_entropy', 'rand_training', 'rand_target']:
        if sample_condition is not None:
            pass
        else:
            entropy_data_mols = ref_mols
            entropy_data_index = list(range(len(entropy_data_mols)))
    
            # print("entropy_data_index shape:", len(entropy_data_index))
            # print("entropy_data_mols shape:", len(entropy_data_mols))
            _, mus, _ = vae.calc_mems(entropy_data_index, entropy_data_mols, save=True)

        if sample_mode in ['high_entropy', 'k_high_entropy']:
            # check entropy of each dimension in mus
            vae_entropy = calc_entropy(mus)
            print("vae_entropy", vae_entropy)
            if save_path is not None:
                with open(save_path.replace('.csv', '_entropy.txt'), 'w') as f:
                    for i, ent in enumerate(vae_entropy):
                        f.write(f"Dimension {i}: {ent}\n")

        elif sample_mode in ['rand_training', 'rand_target']:
            embedding_train = mus
            # print("embedding_train shape:", embedding_train.shape)

    if sample_mode == 'high_entropy':
        # select dimensions with entropy above cutoff
        sample_dims = np.where(np.array(vae_entropy) > entropy_cutoff)[0]
    elif sample_mode == 'k_high_entropy':
        # select top k dimensions with highest entropy
        # sample_dims = np.argpartition(np.array(vae_entropy), -k_entropy)[-k_entropy:]
        sample_dims = np.argsort(vae_entropy)[-k_entropy:]
    elif sample_mode == 'rand':
        # sample from all dimensions
        sample_dims = None
    elif sample_mode == 'rand_training':
        # min max for each dimension in embedding
        min_val = np.min(embedding_train, axis=0)
        max_val = np.max(embedding_train, axis=0)
        sample_dims = list(zip(np.floor(min_val), np.ceil(max_val)))
    elif sample_mode == 'rand_target':
        # sample from training molecules in embedding space
        sample_dims = mus
    else:
        raise ValueError("Invalid sample mode")

    print("sample_mode:", sample_mode)
    print("sample_dims:", sample_dims)
    print("prompt:", prompt)
    ### Generate samples
    start_time = perf_counter()
    samples = []
    for _ in tqdm(range(n_samples), desc="Sampling"):
        if "*" in str(prompt):
            if sample_condition is not None:
                pass
            else: 
                model_sampler = partial(
                    vae.sample, 
                    n=1,
                    method=decode_method,
                    sample_mode=sample_mode,
                    sample_dims=sample_dims,
                    k_entropy=k_entropy,
                    limit_max_len=limit_max_len, 
                    temperature=temperature, 
                    top_k=top_k, 
                    do_sample=do_sample,
                )
            current_samples = xprompt_sampler(model_sampler=model_sampler, 
                                              start_prompt=prompt,
                                              n_samples=n_samples_per_batch, 
                                              limit_gen_size=150,
                                              dummy_attaches_enabled=dummy_attaches_enabled,
                                              seed=seed)
        elif sample_condition:
            pass
        else:
            current_samples = vae.sample(n_samples_per_batch,
                                         method=decode_method,
                                        sample_mode=sample_mode,
                                        sample_dims=sample_dims,
                                        k_entropy=k_entropy,
                                        temperature=temperature,
                                        top_k=top_k,
                                        do_sample=do_sample,
                                        prompt=prompt)
        samples.extend(current_samples)
    stop_time = perf_counter()
    total_time = round(stop_time - start_time, 5)

    if save_path is not None:
        with open(save_path, 'w') as f:
            for sample in samples:
                f.write(sample + '\n')

        mols = []
        samples_clean = [cleaned for smi in samples if (cleaned := cleaner(smi, max_len=limit_max_len)) is not None]
        samples_30 = list(set(samples_clean))[:30] if len(samples_clean) > 30 else list(set(samples_clean))
        for smi in samples_30:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
            else:
                mol = Chem.MolFromSmiles(smi, sanitize=False)
                if mol is not None:
                    mols.append(mol)
        if len(mols) > 0:
            display_molecule(mols)
            plt.savefig(save_path.replace('.csv', '.png'))
        else:
            print("No valid molecules generated, skipping PNG generation.")

    if metrics:
        # valid, unique, novel
        if sample_condition is not None:
            pass
        else:
            smiles_train = list(ref_mols) if ref_mols is not None else []
        smiles_gen = samples
        print("Total generated:", len(smiles_gen), "(100%)")
        print("Total expected:", n_samples * n_samples_per_batch)
        expected_smiles_gen = n_samples * n_samples_per_batch
        assert len(smiles_gen) >= expected_smiles_gen, f"Expected {expected_smiles_gen} samples, but got {len(smiles_gen)}"

        smiles_valid, smiles_invalid = valid(smiles_gen, return_invalid=True)
        valid_percentage = len(smiles_valid) / len(smiles_gen) if len(smiles_gen) > 0 else 0
        valid_percentage = valid_percentage * 100
        print("Valid generated:", len(smiles_valid), f"({valid_percentage:.4}%)")
        smiles_unique = unique(smiles_valid)
        unique_percentage = len(smiles_unique) / len(smiles_gen) if len(smiles_gen) > 0 else 0
        unique_percentage = unique_percentage * 100
        unique_percentage_valid = len(smiles_unique) / len(smiles_valid) if len(smiles_valid) > 0 else 0
        unique_percentage_valid = unique_percentage_valid * 100
        print("Valid&Unique generated:", len(smiles_unique), f"({unique_percentage:.4}%)", f"({unique_percentage_valid:.4}% of valid)")
        smiles_novel = novel(smiles_unique, smiles_train)
        novel_percentage = len(smiles_novel) / len(smiles_gen) if len(smiles_gen) > 0 else 0
        novel_percentage = novel_percentage * 100
        novel_percentage_validunique = len(smiles_novel) / len(smiles_unique) if len(smiles_unique) > 0 else 0
        novel_percentage_validunique = novel_percentage_validunique * 100
        print("Valid&Unique&Novel generated:", len(smiles_novel), f"({novel_percentage:.4}%)", f"({novel_percentage_validunique:.4}% of valid&unique)")

        dictionary = get_fingerprint_dictionary(list(smiles_valid) + list(smiles_train))
        smiles_intdiv = internal_diversity(smiles_valid, dictionary=dictionary) # only valid
        smiles_snn = similarity_to_nearest_neighbor(smiles_valid, smiles_train, dictionary=dictionary) # only valid
        print("Internal diversity (average pairwise Tanimoto distance):", smiles_intdiv)
        print("Similarity to nearest neighbor in training set (average Tanimoto similarity):", smiles_snn)

        if save_path is not None:
            with open(save_path.replace('.csv', '_metrics.txt'), 'w') as f:
                f.write(f"Prompt\t{prompt}\n")
                f.write(f"Total\t{len(smiles_gen)}\t{100.0:.4}%\n")
                f.write(f"Valid\t{len(smiles_valid)}\t{valid_percentage:.4}%\n")
                f.write(f"Unique (to total)\t{len(smiles_unique)}\t{unique_percentage:.4}%\n")
                f.write(f"Unique (to valid)\t{len(smiles_unique)}\t{unique_percentage_valid:.4}%\n")
                f.write(f"Novel (to total)\t{len(smiles_novel)}\t{novel_percentage:.4}%\n")
                f.write(f"Novel (to valid&unique)\t{len(smiles_novel)}\t{novel_percentage_validunique:.4}%\n")
                f.write(f"IntDiv\t\t{smiles_intdiv}\n")
                f.write(f"SNN\t\t{smiles_snn}\n")
                f.write(f"Run Time\t\t{total_time}\n")
                # f.write(f"Recon\t\t{matches_percentage:.4%}\n")

            with open(save_path.replace('.csv', '_invalid.txt'), 'w') as f:
                for smi in smiles_invalid:
                    f.write(smi + '\n')

    if metrics:
        return samples, {"Prompt": prompt,
                         "Total": (len(smiles_gen), 1),
                         "Valid": (len(smiles_valid), valid_percentage),
                         "Unique_Total": (len(smiles_unique), unique_percentage),
                         "Unique_Valid": (len(smiles_unique), unique_percentage_valid),
                         "Novel_Total": (len(smiles_novel), novel_percentage),
                         "Novel_ValidUnique": (len(smiles_novel), novel_percentage_validunique),
                         "IntDiv": smiles_intdiv,
                         "SNN": smiles_snn,
                         "RunTime": total_time}
    else:
        return samples


def sampling_analysis(smiles_gen, save_path, metrics=False):
    smiles_valid = valid(smiles_gen)
    smiles_unique = unique(smiles_valid)
    print("Total generated (unique):", len(smiles_unique))
    # molecular weight
    mol_weights = [get_mol_weight(smi) for smi in smiles_unique]
    mol_weights_avg = np.mean(mol_weights)
    mol_weights_std = np.std(mol_weights)
    print(f"Molecular weight: {mol_weights_avg:.2f} ± {mol_weights_std:.2f}")
    # number of atoms
    num_atoms = [get_num_atoms(smi) for smi in smiles_unique]
    num_atoms_avg = np.mean(num_atoms)
    num_atoms_std = np.std(num_atoms)
    print(f"Number of atoms: {num_atoms_avg:.2f} ± {num_atoms_std:.2f}")
    # sascore
    sascores = [get_sa_score(smi) for smi in smiles_unique]
    sascores_avg = np.nanmean(sascores)
    sascores_std = np.nanstd(sascores)
    print(f"SAScore: {sascores_avg:.2f} ± {sascores_std:.2f}")
    # save
    with open(save_path.replace('.csv', '_analysis.txt'), 'w') as f:
        f.write(f"Total (unique)\t{len(smiles_unique)}\n")
        f.write(f"Molecular weight\t{mol_weights_avg:.2f}\t{mol_weights_std:.2f}\n")
        f.write(f"Number of atoms\t{num_atoms_avg:.2f}\t{num_atoms_std:.2f}\n")
        f.write(f"SAScore\t{sascores_avg:.2f}\t{sascores_std:.2f}\n")
    # plot
    plt.figure(figsize=(15, 4), dpi=300)
    
    plt.subplot(1, 3, 1)
    df = pd.DataFrame({'SMILES': smiles_unique, 'MW': mol_weights, 'NumAtoms': num_atoms, 'SAScore': sascores})
    bin_range = range(0, int(df['MW'].max()) + 10, 100)
    ax = sns.histplot(data=df, x='MW', color=COLOR_MAIN, bins=bin_range, kde=False, stat='density')
    plt.axvline(mol_weights_avg, color='grey', linestyle='--')
    plt.title('Molecular Weight Distribution')
    ax.set_xticks(range(0, int(df['MW'].max()) + 10, 500))
    
    plt.subplot(1, 3, 2)
    bin_range = range(0, df['NumAtoms'].max() + 1, 5)
    ax = sns.histplot(data=df, x='NumAtoms', color=COLOR_MAIN, bins=bin_range, kde=False, stat='density')
    plt.axvline(num_atoms_avg, color='grey', linestyle='--')
    plt.title('Number of Atoms Distribution')
    ax.set_xticks(range(0, int(df['NumAtoms'].max()) + 1, 20))
    
    plt.subplot(1, 3, 3)
    bin_range = np.arange(0, 10.5, 0.5)
    ax = sns.histplot(data=df, x='SAScore', color=COLOR_MAIN, bins=bin_range, kde=False, stat='density')
    plt.axvline(sascores_avg, color='grey', linestyle='--')
    plt.title('SAScore Distribution')
    ax.set_xticks(np.arange(0, 11, 1))
    plt.tight_layout()
    plt.savefig(save_path.replace('.csv', '_analysis.png'))

    if metrics:
        return {"Total_Unique": len(smiles_unique),
                "MW": (mol_weights_avg, mol_weights_std),
                "NumAtoms": (num_atoms_avg, num_atoms_std),
                "SAScore": (sascores_avg, sascores_std)}
    return None

