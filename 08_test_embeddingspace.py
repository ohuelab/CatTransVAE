import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from transvae.parsers import device_init, model_init, sample_parser_mol

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(font="Helvetica", style="ticks", rc=custom_params)

colors = ["#5971C5", "#F29344", "#F34D6F", "#458B73", "#F1C54E", "#7955BD", "#5DA6C5", "#96606B", "#ADCA63", "#AE2A2A", "#7D7330"]
black = "#595A5A"

custom_order = [
             "OSCAR_SEED",
             "OSCAR_DHBD",
             "OSCAR_NHC",
             "ReaLigand",
             "CLC-DB",
             "Kraken",
             "TEPid",
             "TMC_CSD",
             "TMC_TMQMG",
             "tmQMg-L",
             "ORD",
             "PubChem",
             ]


def make_palette(sources):
    unique = list(sources)
    pal = sns.color_palette(colors, n_colors=len(unique)) if len(unique) > 0 else []
    return {s: c for s, c in zip(unique, pal)}


def embeddingspace(args):
    print("Training with args:", args)
    device = device_init(args)
    max_per_source = 1000  # override to limit samples for visualization
    out_dir = 'results/embedding_space'
    save_name = args.checkpoint.split('/')[-1].replace('.pt', '')

    csv_path = os.path.join("data/CatalystSet_TMC_D/CatalystSet_TMC_D.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}. Please place the CSV at this path.")
    csv_path_pubchem = os.path.join("data/pubchem/pubchem_1M.csv")

    df = pd.read_csv(csv_path)
    df_pubchem = pd.read_csv(csv_path_pubchem)
    df_pubchem = df_pubchem[:1000]
    df_pubchem['source'] = 'PubChem'
    df_pubchem.rename(columns={'smiles': 'SMILES'}, inplace=True)
    df = pd.concat([df, df_pubchem], ignore_index=True)

    # sample up to `max_per_source` per source
    rng = np.random.RandomState(args.seed)
    sampled_frames = []
    for s, g in df.groupby('source'):
        if len(g) <= max_per_source:
            sampled_frames.append(g)
        else:
            sampled_frames.append(g.sample(n=max_per_source, random_state=rng))
    df_sample = pd.concat(sampled_frames).reset_index(drop=True)

    # print size if each source
    print("Sampled dataset size by source:")
    print(df_sample['source'].value_counts())

    os.makedirs(out_dir, exist_ok=True)

    # load model
    print('Loading model...')
    # minimal args required by model_init: ensure attributes exist
    vae = model_init(args, mode='inference', load_fn=args.checkpoint)
    vae.model.eval()

    print('Calculating mu vectors from model...')
    data_idxs = np.arange(len(df_sample))
    data_mols = df_sample['SMILES'].to_numpy()
    mems, mus, logvars = vae.calc_mems(data_idxs, data_mols, save=False)

    # mus: (N, d_latent)
    X = mus

    # compute PCA to reduce dimensionality before t-SNE (recommended for high-dim vectors)
    print('Running PCA (50 dims)')
    pca = PCA(n_components=50, random_state=args.seed)
    X_pca = pca.fit_transform(X)

    # t-SNE
    print('Running t-SNE (this may take a while)')
    tsne = TSNE(n_components=2, init='pca', perplexity=30, learning_rate=200, n_iter=1000, random_state=args.seed)
    X_tsne = tsne.fit_transform(X_pca)
    X_tsne_df = pd.DataFrame({
        'tsne_1': X_tsne[:, 0],
        'tsne_2': X_tsne[:, 1],
        'source': df_sample['source']
    })

    # UMAP (import lazily to provide helpful error)
    try:
        import umap.umap_ as umap
    except Exception as e:
        raise ImportError('UMAP not installed. Install with `pip install umap-learn`.') from e

    print('Running UMAP')
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=args.seed)
    X_umap = reducer.fit_transform(X)
    X_umap_df = pd.DataFrame({
        'umap_1': X_umap[:, 0],
        'umap_2': X_umap[:, 1],
        'source': df_sample['source']
    })

    # prepare palette
    sources = df_sample['source'].astype(str).tolist()
    sources_order = sorted(list(set(sources)), key=lambda x: custom_order.index(x) if x in custom_order else len(custom_order))
    pal = {s: c for s, c in zip(sources_order, sns.color_palette(colors, n_colors=len(set(sources))))}
    pal["PubChem"] = "black"

    # Plot t-SNE
    print('Plotting t-SNE')
    plt.figure(figsize=(10, 8), dpi=300)
    sns.scatterplot(
        x='tsne_1',
        y='tsne_2',
        hue='source',
        hue_order=custom_order,
        palette=pal,
        data=X_tsne_df,
        legend="full",
        alpha=0.7,
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title="Source")
    out_tsne = os.path.join(out_dir, f'{save_name}_embedding_tsne.pdf')
    plt.tight_layout()
    plt.savefig(out_tsne, dpi=300, format='pdf', bbox_inches='tight')
    plt.close()

    # Plot UMAP
    print('Plotting UMAP')
    plt.figure(figsize=(10, 8), dpi=300)

    out_umap = os.path.join(out_dir, f'{save_name}_embedding_umap.pdf')
    sns.scatterplot(
        x='umap_1',
        y='umap_2',
        hue='source',
        hue_order=custom_order,
        palette=pal,
        data=X_umap_df,
        legend="full",
        alpha=0.7,
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title="Source")
    plt.tight_layout()
    plt.savefig(out_umap, dpi=300, format='pdf', bbox_inches='tight')
    plt.close()

    # silhouette scores (on original mus projected space or on embeddings?)
    print('Calculating silhouette scores')
    label_to_int = {s: i for i, s in enumerate(set(sources))}
    labels = np.array([label_to_int[x] for x in sources])

    scores = {}
    if len(set(sources)) > 1:
        try:
            scores['tsne'] = silhouette_score(X_tsne, labels)
        except Exception:
            scores['tsne'] = np.nan
        try:
            scores['umap'] = silhouette_score(X_umap, labels)
        except Exception:
            scores['umap'] = np.nan
        try:
            scores['mu'] = silhouette_score(X, labels)
        except Exception:
            scores['mu'] = np.nan
    else:
        scores['tsne'] = np.nan
        scores['umap'] = np.nan
        scores['mu'] = np.nan

    print('Silhouette scores:', scores)
    with open(os.path.join(out_dir, f'{save_name}_silhouette_scores.txt'), 'w') as f:
        for k, v in scores.items():
            f.write(f"{k}: {v}\n")


if __name__ == '__main__':
    parser = sample_parser_mol()
    args = parser.parse_args()
    embeddingspace(args)

