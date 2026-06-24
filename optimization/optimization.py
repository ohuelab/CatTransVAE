from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import pickle
import torch
from functools import partial
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

import joblib
from xgboost import XGBRegressor
sys.path.append("/CatTransVAE/")
from transvae.parsers import device_init, optimization_parser_mol
from transvae.sampling import sampling
from transvae.training_mol import TransVAE
from transvae.tvae_util import *
from xpromptsmiles.xprompt import *

# Set font to Helvetica
# plt.rcParams["font.sans-serif"] = ["Helvetica"]
plt.rcParams["font.family"] = "sans-serif"

# Alternatively, set directly in sns.set_theme
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(font="Helvetica", style="ticks", rc=custom_params)
sns.set_theme(style="ticks", rc=custom_params)

colors_main = ["#E37185"]
colors = ["#5971C5", "#889ADB", "#B7C6F9", "#E1E8FF"]
light_black = "#595A5A"
black = "#1F1F1F"

def optimize(args):
    print("Training with args:", args)
    device = device_init(args)

    seed = args.seed

    # set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pred_data = args.prediction_dataset
    pred_emb = args.prediction_embeddings
    ckp_gen_model = args.checkpoint_gen.split("/")[-1].replace(".pt", "")
    ckp_pred_model = args.checkpoint_pred
    save_name = args.save_name
    save_name = f"{ckp_pred_model}/opt_{seed}_{save_name}"

    ### Dataset metadata
    _dataset = pd.read_csv(os.path.join("prediction", "datasets", "_dataset.csv"))
    _dataset_meta = _dataset[_dataset["dataset"] == pred_data].iloc[0]
    _dataset_X = _dataset_meta["X"]
    _dataset_y = _dataset_meta["y"]
    _dataset_split = _dataset_meta["split"]
    _dataset_split_train = _dataset_meta["split_train"]
    _dataset_split_val = _dataset_meta["split_val"]
    _dataset_split_test = _dataset_meta["split_test"]
    print(f"Dataset: {_dataset_meta['dataset']}")
    print(f"X column: {_dataset_X}, y column: {_dataset_y}, split column: {_dataset_split}")
    print(f"Train split value: {_dataset_split_train}, Val split value: {_dataset_split_val}, Test split value: {_dataset_split_test}")
    print("#############################")
    print()
    

    df = pd.read_csv(os.path.join("prediction", "datasets", pred_data))
    smiles = df[_dataset_X].values

    # Generative model
    if "CatTransVAE" in args.prediction_embeddings:
        ### Test model
        print("### Loading model...")
        ckpt_gen_path = args.checkpoint_gen
        vae = TransVAE(args, mode="inference", load_fn=ckpt_gen_path)
        print("Loaded best model from:", ckpt_gen_path)
        print("Best val loss:", vae.best_loss)
        print("Best epoch:", vae.best_epoch)
        vae.model.eval()
        print("#############################")
        print()

    # Predictive model
    # load scaler
    with open(os.path.join(ckp_pred_model, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    # load xgboost model
    model_path = Path(ckp_pred_model, "xgboost_best_5fold_model.joblib")
    best_model = joblib.load(model_path)
    print("best_model:", best_model)

    ### Make directory
    # save_name = f"{args.seed}_{args.save_name}"
    os.makedirs(os.path.join(save_name), exist_ok=True)


    ### Sampling after training
    print("### Testing sampling...")
    train_mols_sample = [cleaner(s, max_len=300) for s in smiles]
    ref_mols = train_mols_sample

    sample_mode = args.sample_mode
    decode_method = args.decode_method
    entropy_cutoff = args.entropy_cutoff
    k_entropy = args.k_entropy
    temperature = args.temperature
    top_k = args.top_k
    do_sample = str(args.do_sample).lower() == 'true' or args.do_sample == True
    n_samples = args.n_samples
    n_samples_per_batch = args.n_samples_per_batch
    # total samples = n_samples * n_samples_per_batch
    dummy_attaches_enabled = str(args.dummy_attaches_enabled).lower() == 'true' or args.dummy_attaches_enabled == True
    prompt = args.prompt
    if 'none' not in str(args.prompt):
        prompt = args.prompt.split(',')
    if prompt is None or 'none' in str(prompt):
        prompt = None
    elif "*" in str(prompt) and "," not in str(prompt):
        prompt = prompt[0]
    elif 'none' not in str(prompt):
        prompt = prompt[0]
    limit_max_len = 300
    latent_dim = vae.d_latent

    if sample_mode in ['high_entropy', 'k_high_entropy', 'rand_training', 'rand_target']:
        entropy_data_mols = ref_mols
        entropy_data_index = list(range(len(entropy_data_mols)))

        # print("entropy_data_index shape:", len(entropy_data_index))
        # print("entropy_data_mols shape:", len(entropy_data_mols))
        _, mus, _ = vae.calc_mems(entropy_data_index, entropy_data_mols, save=True)

        if sample_mode in ['high_entropy', 'k_high_entropy']:
            vae_entropy = calc_entropy(mus)
            # print("vae_entropy", vae_entropy)
        elif sample_mode in ['rand_training', 'rand_target']:
            embedding_train = mus
            # print("embedding_train shape:", embedding_train.shape)
    
    if sample_mode == 'high_entropy':
        # select dimensions with entropy above cutoff
        sample_dims = np.where(np.array(vae_entropy) > entropy_cutoff)[0]
    elif sample_mode == 'k_high_entropy':
        # select top k dimensions with highest entropy
        # sample_dims = np.argpartition(np.array(vae_entropy), -k_entropy)[-k_entropy:]
        print("vae_entropy:", vae_entropy)
        sample_dims = np.argsort(vae_entropy)[-k_entropy:]
        
        # dimension_constraint
        dimension_train =[(-2.5, 2.5)] * latent_dim
        d_select = np.random.choice(sample_dims, size=k_entropy, replace=False)
        # print("Selected dimensions:", d_select)
        # loop not in d_select, set to 0
        # z = torch.randn((size, self.d_latent), device=device)
        for d in range(vae.d_latent):
            if d not in d_select:
                # z_opt[:,d] = 0
                dimension_train[d] = [0.0]

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

    print("Sample mode:", sample_mode)
    print("Sample dims:", sample_dims)
    print("Decode method:", decode_method)
    print("Entropy cutoff:", entropy_cutoff)
    print("k_entropy:", k_entropy)
    print("Temperature:", temperature)
    print("Top-k:", top_k)
    print("Do sample:", do_sample)
    print("Prompt:", prompt)
    print("#############################")

    from skopt import gp_minimize


    def sampler_reduced(z_opt, vae, sample_mode, sample_dims, k_entropy, decode_method, limit_max_len, temperature, top_k, do_sample, prompt=None):
        # z_opt = torch.tensor(z_opt, dtype=torch.float32, device=device).unsqueeze(0)
        # print("z_opt", z_opt)
        mode = sample_mode
        if mode == 'rand':
            z = z_opt
        elif mode == 'rand_training':
            low = torch.tensor([d[0] for d in sample_dims], dtype=torch.float)
            high = torch.tensor([d[1] for d in sample_dims], dtype=torch.float)
            z = low + (high - low) * z_opt
        elif mode == 'rand_target':
            sample_dims_list = [torch.tensor(d, dtype=torch.float) for d in sample_dims]
            z = random.choices(sample_dims_list, k=size)[0].unsqueeze(0)
            noise = z_opt * 0.5
            z = z + noise
        elif mode == 'k_high_entropy':
            # print("z_opt before masking:", z_opt)
            # z = torch.zeros((size, self.d_latent))
            d_select = np.random.choice(sample_dims, size=k_entropy, replace=False)
            # print("Selected dimensions:", d_select)
            # loop not in d_select, set to 0
            # z = torch.randn((size, self.d_latent), device=device)
            for d in range(vae.d_latent):
                if d not in d_select:
                    z_opt[:,d] = 0
            z = z_opt
            # print("z_opt after masking:", z_opt)

        # print("prompt:", prompt)
        condition = tokenizer(prompt) if prompt is not None else []
        # print("Condition:", condition)
        mem = z
        # print("MEM:", mem)
        
        ### Decode logic
        if decode_method == 'greedy':
            decoded = vae.greedy_decode(mem, condition=condition, 
                                        limit_max_len=limit_max_len, 
                                        temperature=temperature, 
                                        top_k=top_k, 
                                        do_sample=do_sample)
        elif decode_method == 'beam':
            decoded = vae.beam_decode(mem, condition=condition, 
                                    beam_width=10, limit_max_len=limit_max_len)
        else:
            decoded = None
        
        if decode_method == 'greedy':
            decoded = decode_mols(decoded, vae.org_dict)
        elif decode_method == 'beam':
            decoded_all = []
            for d in decoded:
                decoded_all.extend([decode_mols(torch.tensor(tokens[1:]).unsqueeze(0), vae.org_dict)[0] for tokens in d][0:1])
            decoded = decoded_all

        return decoded
    
    record_results = []

    def objective(z_opt, output=False):
        
        z_opt = torch.tensor(z_opt, dtype=torch.float32, device=device).unsqueeze(0) # shape (1, latent_dim)

        with torch.no_grad():
            if prompt is None:
                ### Decode logic
                decoded = sampler_reduced(z_opt, vae, sample_mode, sample_dims, k_entropy, decode_method, limit_max_len, temperature, top_k, do_sample, prompt=None)

                data_mols = [s for s in decoded if s is not None]
                data_idxs = list(range(len(data_mols)))
            
            else:
                model_sampler_reduced = partial(
                    sampler_reduced, 
                    z_opt=z_opt,
                    vae=vae, 
                    sample_mode=sample_mode, 
                    sample_dims=sample_dims, 
                    k_entropy=k_entropy, 
                    decode_method=decode_method, 
                    limit_max_len=limit_max_len, 
                    temperature=temperature, 
                    top_k=top_k, 
                    do_sample=do_sample
                )
                # print("[OBJ] prompt:", prompt)
                decoded = xprompt_sampler(model_sampler=model_sampler_reduced, 
                                                start_prompt=prompt,
                                                n_samples=n_samples_per_batch, 
                                                limit_gen_size=150,
                                                dummy_attaches_enabled=dummy_attaches_enabled,
                                                seed=seed)

                data_mols = [s for s in decoded if s is not None]
                data_idxs = list(range(len(data_mols)))

            if len(data_mols) == 0:
                return 100
            
            _, mu, _, embs, masks = vae.calc_mems(data_idxs, data_mols, save_dir='memory', save_fn='model_name', return_embedded=True, save=False)

            
            def mean_pooling(embeddings, masks):
                masks = masks.squeeze(1)
                masked_embeddings = embeddings * masks[:, :, None]
                sum_embeddings = masked_embeddings.sum(axis=1)
                count_non_zero = masks.sum(axis=1)[:, None]
                mean_embeddings = sum_embeddings / np.maximum(count_non_zero, 1)
                return mean_embeddings
            
            mean_embeddings = mean_pooling(embs, masks)
            if pred_emb == "CatTransVAE":
                feat = np.concatenate([mu, mean_embeddings], axis=1)

            raw_score = best_model.predict(scaler.transform(feat))
            raw_score = raw_score.item()
            # score must be max at -27.55
            score = np.abs(raw_score - -27.55)
            # score = np.abs(score - 100)
            # panelty for large molecules
            decoded_mol = Chem.MolFromSmiles(decoded[0])
            # check P has 3 neighbors
            if decoded_mol is not None:
                atom_P = [atom for atom in decoded_mol.GetAtoms() if atom.GetSymbol() == 'P']
                if len(atom_P) > 0:
                    for P_i in atom_P:
                        num_neighbors = P_i.GetDegree()
                        if num_neighbors != 3:
                            score += 50
                            break
            if len(decoded[0]) > 100:
                score += 10

            record_results.append({
                "z_opt": z_opt.cpu().numpy(),
                "decoded": decoded,
                "raw_score": raw_score,
                "score": score
            })

        if output:
            return decoded, score, raw_score
        return score

    size = 1

    result = gp_minimize(
        func=objective,
        # dimensions=[(-1.0, 1.0)] * latent_dim,
        dimensions=dimension_train,
        n_calls=200,
        n_initial_points=50,
        acq_func="gp_hedge",
        random_state=seed
    )

    # best latent vector
    best_z = torch.tensor(result.x, dtype=torch.float32, device=device).unsqueeze(0)

    # decode best sample
    smiles, best_score, raw_score = objective(result.x, output=True)
    
    print("Best raw score:", raw_score)
    print("Best score:", best_score)
    print("Best latent:", result.x)
    print("Best sample:", smiles)

    results_df = pd.DataFrame({
        "iteration": [i for i in list(range(len(record_results)))],
        "score": [r["score"] for r in record_results],
        "smiles": [r["decoded"][0] for r in record_results],
        "raw_scores": [r["raw_score"] for r in record_results],
    })
    results_df.to_csv(os.path.join(save_name,
                                'optimization_results.csv'), index=False)
    
    # # save results, round, smiles, score.
    smiles, scores, raw_scores = [], [], []
    for x in result.x_iters:
        s, score, raw_score = objective(x, output=True)
        smiles.append(s[0])
        scores.append(score)
        raw_scores.append(raw_score)

    results_df_again = pd.DataFrame({
        "iteration": [i for i in list(range(len(result.x_iters)))],
        "score": scores,
        "smiles": smiles,
        "raw_scores": raw_scores,
    })
    results_df_again.to_csv(os.path.join(save_name,
                                'optimization_results_again.csv'), index=False)

    # plot graph of optimization
    import matplotlib.pyplot as plt
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(font="Helvetica", style="ticks", rc=custom_params)
    colors = ["#5971C5", "#F29344", "#F34D6F", "#458B73", "#F1C54E", "#7955BD", "#5DA6C5", "#96606B", "#ADCA63", "#AE2A2A", "#7D7330"]
    black = "#595A5A"
    plt.plot(result.func_vals, color=colors[0], marker='o', linestyle='-')
    plt.xlabel("Iteration")
    plt.ylabel("Negative Score")
    plt.title("Bayesian Optimization of Latent Space")
    plt.savefig(os.path.join(save_name,
                        'optimization_plot.png'))
    plt.close()

    # show top 10 smiles and scores
    unique_df = results_df.drop_duplicates(subset=["smiles"])
    top_results = unique_df.sort_values(by="score", ascending=True).head(30)
    mols = [Chem.MolFromSmiles(s) for s in top_results["smiles"]]
    top_results["mols"] = mols
    Draw.MolsToGridImage(top_results["mols"].tolist(), legends=[f"{s[:10]}\n{score:.2f}" for s, score in zip(top_results["smiles"], top_results["score"])], molsPerRow=5, subImgSize=(250,250)).save(os.path.join(save_name, 
                        'top_results.png'))
    
    # save optimization parameters
    with open(os.path.join(save_name, 'optimization_params.txt'), 'w') as f:
        f.write(f"Seed: {seed}\n")
        f.write(f"Prediction dataset: {pred_data}\n")
        f.write(f"Prediction embeddings: {pred_emb}\n")
        f.write(f"Checkpoint generative model: {ckp_gen_model}\n")
        f.write(f"Checkpoint predictive model: {ckp_pred_model}\n")
        f.write(f"Sample mode: {sample_mode}\n")
        f.write(f"Decode method: {decode_method}\n")
        f.write(f"Entropy cutoff: {entropy_cutoff}\n")
        f.write(f"k_entropy: {k_entropy}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Top-k: {top_k}\n")
        f.write(f"Do sample: {do_sample}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Number of samples per batch: {n_samples_per_batch}\n")
        f.write(f"Dummy attaches enabled: {dummy_attaches_enabled}\n")
        f.write(f"Total samples: {n_samples * n_samples_per_batch}\n")
        f.write(f"Best score: {best_score}\n")
        f.write(f"Best latent vector: {result.x}\n")
        f.write(f"n_calls: {len(result.func_vals)}\n")
        f.write(f"n_initial_points: {150}\n")
        f.write(f"Acquisition function: gp_hedge\n")

    # plot histogram of scores
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    if 'suzuki' in pred_data.lower():
        bins = np.arange(-80, 40, 1)
        # original data
        if 'Pd' in prompt:
            df_Pd = df[df[_dataset_X].str.contains("Pd")]
            df_Pd_y = df_Pd[_dataset_y].values
            hist2 = sns.kdeplot(list(df_Pd_y), bw_adjust=0.8, color=colors[0], alpha=0.2, label="Dataset complexes", fill=True)
        hist1 = sns.kdeplot(list(results_df["raw_scores"]), bw_adjust=0.8, color=colors_main[0], alpha=0.5, label="Optimized complexes", fill=True)
        # highlight optimization range as range in graph (−32.1 to −23.0) as light grey area
        ax.axvspan(-32.1, -23.0, color=light_black, alpha=0.2, label="Target region")
    ax.set_xlabel("Dataset and predicted bininding energy (kcal mol$^{-1}$)")
    ax.set_ylabel("Density")
    # ax.set_title("Distribution of dataset and predicted binding energy during optimization")
    handles, labels = plt.gca().get_legend_handles_labels()
    order = ['Dataset complexes', 'Optimized complexes', 'Target region']
    handles_labels = dict(zip(labels, handles))
    ordered_handles = [handles_labels[label] for label in order]
    # plt.legend(ordered_handles, order)
    # legend outside of plot on the right
    plt.legend(ordered_handles, order, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(os.path.join(save_name, 'optimization_histogram.pdf'), bbox_inches='tight')
    plt.close()

    return

if __name__ == '__main__':
	parser = optimization_parser_mol()
	args = parser.parse_args()
	optimize(args)
