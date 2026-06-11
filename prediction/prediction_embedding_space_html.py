import os
import sys
import pickle
import io
import base64
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

sys.path.append("/gs/bs/tga-ohuelab/kengkanna/CatXPro/")
from transvae.sampling import reconstructing, sampling
from transvae.training_mol import TransVAE
from transvae.parsers import device_init, model_init, prediction_parser_mol
from transvae.tvae_util import *

import json
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Set font to Helvetica
# plt.rcParams["font.sans-serif"] = ["Helvetica"]
plt.rcParams["font.family"] = "sans-serif"

# Alternatively, set directly in sns.set_theme
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(font="Helvetica", style="ticks", rc=custom_params)
sns.set_theme(style="ticks", rc=custom_params)


from typing import Iterable, Optional, Sequence, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _choose_pca_dim(n_samples: int, n_features: int, max_components: int = 50) -> int:
	if n_samples < 3:
		raise ValueError("At least 3 samples are required to compute a 2D embedding.")
	return max(2, min(max_components, n_samples - 1, n_features))


def _choose_tsne_perplexity(n_samples: int) -> float:
	# t-SNE requires perplexity < n_samples. A moderate heuristic works well for ~15k points.
	if n_samples < 5:
		raise ValueError("At least 5 samples are required for t-SNE.")
	return float(max(5, min(50, (n_samples - 1) // 3)))


def _choose_umap_neighbors(n_samples: int) -> int:
	if n_samples < 3:
		raise ValueError("At least 3 samples are required for UMAP.")
	return int(max(5, min(30, n_samples - 1)))


def _rainbow_cmap():
	return sns.color_palette("rainbow", as_cmap=True)


def _smiles_to_data_uri(smiles: str, size: Tuple[int, int] = (220, 150)) -> Optional[str]:
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _save_interactive_tsne_html(
    points: np.ndarray,
    values: np.ndarray,
    smiles: Sequence[str],
    colorby: Optional[Sequence[str]] = None,
    save_path: str = "embedding_space_tsne_interactive.html",
    title: str = "Embedding space (t-SNE)",
) -> str:
    try:
        import plotly.graph_objects as go
    except Exception as exc:  # pragma: no cover - import error depends on environment
        raise ImportError("Plotly is required for interactive HTML output. Install it with `pip install plotly`.") from exc

    if len(smiles) != len(points):
        raise ValueError("`smiles` length must match number of points for interactive plotting.")

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    if colorby is not None:
        if len(colorby) != len(points):
            raise ValueError("`colorby` length must match number of points.")
        unique_cats = np.unique(colorby)
        cmap = sns.color_palette("husl", n_colors=len(unique_cats))
        cat_to_color = {cat: f"rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})" for cat, c in zip(unique_cats, cmap)}
        c_colorby = [cat_to_color[cat] for cat in colorby]

    fig = go.Figure(
        data=[
            go.Scattergl(
                x=points[:, 0],
                y=points[:, 1],
                mode="markers",
                marker={
                    "size": 7,
                    "opacity": 0.75,
                    "color": c_colorby if colorby is not None else values,
                    "colorscale": "Rainbow",
                    "cmin": vmin,
                    "cmax": vmax,
                    "colorbar": {"title": "Value"},
                    "line": {"width": 0.3, "color": "white"},
                },
                hovertemplate="t-SNE 1: %{x:.3f}<br>t-SNE 2: %{y:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
    )

    records = []
    for s, v in zip(smiles, values):
        records.append(
            {
                "smiles": "" if s is None else str(s),
                "value": None if not np.isfinite(v) else float(v),
                "img": _smiles_to_data_uri(s),
            }
        )

    plot_div_id = "tsne-plot"
    panel_id = "hover-panel"
    html_payload = fig.to_json()
    records_payload = json.dumps(records)

    html_content = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #fafafa; }}
    .wrap {{ display: grid; grid-template-columns: 1fr 300px; gap: 16px; padding: 16px; }}
    #{plot_div_id} {{ min-height: 720px; border: 1px solid #ddd; background: #fff; }}
    #{panel_id} {{
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 12px;
      height: fit-content;
      position: sticky;
      top: 16px;
    }}
    .mol-card img {{ width: 100%; border: 1px solid #eee; border-radius: 6px; background: #fff; }}
    .kv {{ font-size: 13px; margin-top: 8px; line-height: 1.4; word-break: break-all; }}
    .muted {{ color: #666; font-size: 13px; }}
    @media (max-width: 980px) {{
      .wrap {{ grid-template-columns: 1fr; }}
      #{panel_id} {{ position: static; }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\"> 
    <div id=\"{plot_div_id}\"></div>
    <div id=\"{panel_id}\" class=\"mol-card\">
      <div class=\"muted\">Hover a point to view molecule details.</div>
    </div>
  </div>
  <script>
    const fig = {html_payload};
    const records = {records_payload};
    const panel = document.getElementById('{panel_id}');

    function renderCard(rec, idx) {{
      const val = rec.value === null || Number.isNaN(rec.value) ? 'NaN' : rec.value.toFixed(6);
      const imgHtml = rec.img
        ? `<img src=\"${{rec.img}}\" alt=\"molecule\" />`
        : `<div class=\"muted\">Molecule image unavailable.</div>`;
      panel.innerHTML = `
        ${{imgHtml}}
        <div class=\"kv\"><strong>Index:</strong> ${{idx}}</div>
        <div class=\"kv\"><strong>Value:</strong> ${{val}}</div>
        <div class=\"kv\"><strong>SMILES:</strong><br/>${{rec.smiles || ''}}</div>
      `;
    }}

    Plotly.newPlot('{plot_div_id}', fig.data, fig.layout, {{responsive: true}}).then((gd) => {{
      gd.on('plotly_hover', (ev) => {{
        if (!ev || !ev.points || ev.points.length === 0) return;
        const i = ev.points[0].pointIndex;
        renderCard(records[i] || {{}}, i);
      }});
      gd.on('plotly_unhover', () => {{
        panel.innerHTML = '<div class=\"muted\">Hover a point to view molecule details.</div>';
      }});
    }});
  </script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return save_path


def plot_embedding_space_by_value(
    embeddings: np.ndarray,
    values: np.ndarray,
    smiles: Optional[Sequence[str]] = None,
    colorby: Optional[Sequence[str]] = None,
    save_path: str = "plot",
    file_prefix: str = "embedding_space",
    seed: int = 0,
    tsne_perplexity: int = None,
    umap_n_neighbors: int = None,
    umap_min_dist: float = 0.1,
) -> dict:
    """
    Compute 2D t-SNE and UMAP embeddings from arbitrary latent vectors and plot them
    colored by a continuous rainbow spectrum.

    Parameters
    ----------
    embeddings:
        Array-like of shape (n_samples, n_features).
    values:
        Scalar value per sample. The values are mapped to a continuous rainbow spectrum.
    save_path:
        Directory where PDF plots are written.
    file_prefix:
        Prefix for the saved files.
    seed:
        Random seed for PCA, t-SNE, and UMAP.
    dpi:
        Figure resolution when saving.
    tsne_perplexity:
        Optional perplexity override. If omitted, a data-dependent heuristic is used.
    umap_n_neighbors:
        Optional UMAP neighbor override. If omitted, a data-dependent heuristic is used.
    umap_min_dist:
        UMAP minimum distance.
    show:
        If True, display plots interactively in addition to saving them.

    Returns
    -------
    dict
        Dictionary containing the 2D embeddings, bin assignments, and saved paths.
    """
    X = np.asarray(embeddings)
    y = np.asarray(values).reshape(-1)
    colorby = None if colorby is None else np.asarray(colorby).reshape(-1)
    smiles_arr = None if smiles is None else np.asarray(smiles).reshape(-1)

    if X.shape[0] != y.shape[0]:
        raise ValueError("`embeddings` and `values` must contain the same number of samples.")
    if smiles_arr is not None and X.shape[0] != smiles_arr.shape[0]:
        raise ValueError("`smiles` and `embeddings` must contain the same number of samples.")

    n_samples, n_features = X.shape
    pca_dim = _choose_pca_dim(n_samples, n_features)
    perplexity = float(tsne_perplexity) if tsne_perplexity is not None else _choose_tsne_perplexity(n_samples)
    perplexity = min(perplexity, max(2.0, n_samples - 1.0 - 1e-6))
    neighbors = int(umap_n_neighbors) if umap_n_neighbors is not None else _choose_umap_neighbors(n_samples)
    neighbors = max(2, min(neighbors, n_samples - 1))

    # PCA before t-SNE makes the optimization much more stable for 512+ dimensional latents.
    if n_features > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=seed)
        X_for_tsne = pca.fit_transform(X)
    else:
        X_for_tsne = X

    tsne = TSNE(
        n_components=2,
        init="pca",
        perplexity=perplexity,
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    )
    X_tsne = tsne.fit_transform(X_for_tsne)

    try:
        import umap.umap_ as umap
    except Exception as exc:  # pragma: no cover - import error depends on environment
        raise ImportError("UMAP is required for this function. Install it with `pip install umap-learn`.") from exc

    reducer = umap.UMAP(
        n_neighbors=neighbors,
        min_dist=umap_min_dist,
        n_components=2,
        metric="euclidean",
        random_state=seed,
    )
    X_umap = reducer.fit_transform(X)

    finite_values = y[np.isfinite(y)]
    if finite_values.size == 0:
        raise ValueError("`values` must contain at least one finite value.")
    vmin = float(finite_values.min())
    vmax = float(finite_values.max())

    if colorby is not None:
        # color as category
        cmap = sns.color_palette("husl", n_colors=len(np.unique(colorby)))
        c_colorby = {cat: cmap[i] for i, cat in enumerate(np.unique(colorby))}
    else:
        cmap = _rainbow_cmap()

    os.makedirs(save_path, exist_ok=True)
    tsne_path = os.path.join(save_path, f"{file_prefix}_tsne.pdf")
    umap_path = os.path.join(save_path, f"{file_prefix}_umap.pdf")
    tsne_html_path = os.path.join(save_path, f"{file_prefix}_tsne_interactive.html")

    def _plot(points: np.ndarray, title: str, xlabel: str, ylabel: str, save_path: str) -> None:
        # fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=y if colorby is None else [c_colorby[cat] for cat in colorby],
            cmap=cmap if colorby is None else None,
            vmin=vmin,
            vmax=vmax,
            s=20,
            alpha=0.7,
            edgecolors="w",
            linewidths=0.5,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # legend
        legend_elements = []
        if colorby is not None:
            for cat in np.unique(colorby):
                legend_elements.append(plt.Line2D([0], [0], marker="o", color="w", label=str(cat), markerfacecolor=c_colorby[cat], markersize=8))
            ax.legend(handles=legend_elements, title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")

        if colorby is None:
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Value")
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, format="pdf", bbox_inches="tight")
        plt.show()
        plt.close(fig)

    _plot(X_tsne, "Embedding space (t-SNE)", "t-SNE 1", "t-SNE 2", tsne_path)
    _plot(X_umap, "Embedding space (UMAP)", "UMAP 1", "UMAP 2", umap_path)

    if smiles_arr is not None:
        _save_interactive_tsne_html(
            points=X_tsne,
            values=y,
            smiles=smiles_arr,
            save_path=tsne_html_path,
            title="Embedding space (t-SNE, interactive)",
            colorby=colorby
        )

    return {
        "tsne": X_tsne,
        "umap": X_umap,
        "tsne_path": tsne_path,
        "umap_path": umap_path,
        "tsne_html_path": tsne_html_path if smiles_arr is not None else None,
        "pca_dim": pca_dim,
        "tsne_perplexity": perplexity,
        "umap_n_neighbors": neighbors,
    }


def main(args):
    print("Training with args:", args)
    # device = device_init(args)
    seed = args.seed
    pred_data = args.prediction_dataset
    pred_emb = args.prediction_embeddings
    ckp_model = args.checkpoint.split("/")[-1].replace(".pt", "")
    save_name = args.save_name

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

    ### Dataset splitting
    if _dataset_split == "none":
        new_filename = f"{pred_data.split('.')[0]}_split_{seed}.csv"
        if os.path.exists(os.path.join("prediction", "datasets", new_filename)):
            print("Split dataset already exists. Loading from file.")
        else:
            df = pd.read_csv(os.path.join("prediction", "datasets", pred_data))
            df_trainval, df_test = train_test_split(df, test_size=_dataset_split_test, random_state=seed)
            val_size = _dataset_split_val * len(df) / len(df_trainval)
            df_train, df_val = train_test_split(df_trainval, test_size=val_size, random_state=seed)
            df_train['split'] = 'train'
            df_val['split'] = 'val'
            df_test['split'] = 'test'
            df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)


            df_all.to_csv(os.path.join("prediction", "datasets", new_filename), index=False)
        pred_data = new_filename
    else:
        df = pd.read_csv(os.path.join("prediction", "datasets", pred_data))
        new_filename = f"{pred_data.split('.')[0]}_split_{seed}.csv"
        df.to_csv(os.path.join("prediction", "datasets", new_filename), index=False)
        pred_data = new_filename


    if "CatTransVAE" in args.prediction_embeddings:
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
    
    os.makedirs(os.path.join('prediction','embeddings', f"{pred_data}"), exist_ok=True)
    if pred_emb == "CatTransVAE" or pred_emb == "CatTransVAE_emb" or pred_emb == "CatTransVAE_vae":
        os.makedirs(os.path.join('prediction','results', f"{pred_data}", f"{seed}_{pred_emb}_{ckp_model}"), exist_ok=True)
        save_name_embeddings = f"{pred_data}/{seed}_{pred_emb}_{ckp_model}"
        save_name = f"{pred_data}/{seed}_{pred_emb}_{ckp_model}/emb_{save_name}"
    elif pred_emb == "MorganFP":
        os.makedirs(os.path.join('prediction','results', f"{pred_data}", f"{seed}_{pred_emb}"), exist_ok=True)
        save_name_embeddings = f"{pred_data}/{seed}_{pred_emb}"
        save_name = f"{pred_data}/{seed}_{pred_emb}/emb_{save_name}"
    elif pred_emb == "CatTransVAE+MorganFP" or pred_emb == "CatTransVAE_emb+MorganFP" or pred_emb == "CatTransVAE_vae+MorganFP":
        os.makedirs(os.path.join('prediction','results', f"{pred_data}", f"{seed}_{pred_emb}_{ckp_model}"), exist_ok=True)
        save_name_embeddings = f"{pred_data}/{seed}_{pred_emb}_{ckp_model}"
        save_name = f"{pred_data}/{seed}_{pred_emb}_{ckp_model}/emb_{save_name}"
    elif pred_emb == "SMI_TED" or pred_emb == "ChemBERTa" or pred_emb == "MoLFormer" or pred_emb == "ChemBERTa_MLM" or pred_emb == "ChemBERTa_MTR"  or pred_emb == "MIST":
        os.makedirs(os.path.join('prediction','results', f"{pred_data}", f"{seed}_{pred_emb}"), exist_ok=True)
        save_name_embeddings = f"{pred_data}/{seed}_{pred_emb}"
        save_name = f"{pred_data}/{seed}_{pred_emb}/emb_{save_name}"

    if os.path.exists(os.path.join('prediction','embeddings', f"{save_name_embeddings}_train.pkl")) \
        and os.path.exists(os.path.join('prediction','embeddings', f"{save_name_embeddings}_val.pkl")) \
            and os.path.exists(os.path.join('prediction','embeddings', f"{save_name_embeddings}_test.pkl")):
        print("Embeddings already exist. Loading from file.")
        with open(os.path.join('prediction','embeddings', f"{save_name_embeddings}_train.pkl"), 'rb') as f:
            embedding_train = pickle.load(f)
        with open(os.path.join('prediction','embeddings', f"{save_name_embeddings}_val.pkl"), 'rb') as f:
            embedding_val = pickle.load(f)
        with open(os.path.join('prediction','embeddings', f"{save_name_embeddings}_test.pkl"), 'rb') as f:
            embedding_test = pickle.load(f)
    elif pred_emb == "CatTransVAE+MorganFP" or pred_emb == "CatTransVAE_emb+MorganFP" or pred_emb == "CatTransVAE_vae+MorganFP":
        save_name_embeddings_cat = f"{pred_data}/{seed}_CatTransVAE_{ckp_model}"
        save_name_embeddings_morgan = f"{pred_data}/{seed}_MorganFP"
        if os.path.exists(os.path.join('prediction','embeddings', f"{save_name_embeddings_cat}_train.pkl")) \
            and os.path.exists(os.path.join('prediction','embeddings', f"{save_name_embeddings_cat}_val.pkl")) \
                and os.path.exists(os.path.join('prediction','embeddings', f"{save_name_embeddings_cat}_test.pkl")) \
                    and os.path.exists(os.path.join('prediction','embeddings', f"{save_name_embeddings_morgan}_train.pkl")) \
                        and os.path.exists(os.path.join('prediction','embeddings', f"{save_name_embeddings_morgan}_val.pkl")) \
                            and os.path.exists(os.path.join('prediction','embeddings', f"{save_name_embeddings_morgan}_test.pkl")):
            print("Embeddings already exist. Loading from file.")
            with open(os.path.join('prediction','embeddings', f"{save_name_embeddings_cat}_train.pkl"), 'rb') as f:
                embedding_train_cat = pickle.load(f)
            with open(os.path.join('prediction','embeddings', f"{save_name_embeddings_cat}_val.pkl"), 'rb') as f:
                embedding_val_cat = pickle.load(f)
            with open(os.path.join('prediction','embeddings', f"{save_name_embeddings_cat}_test.pkl"), 'rb') as f:
                embedding_test_cat = pickle.load(f)
            with open(os.path.join('prediction','embeddings', f"{save_name_embeddings_morgan}_train.pkl"), 'rb') as f:
                embedding_train_morgan = pickle.load(f)
            with open(os.path.join('prediction','embeddings', f"{save_name_embeddings_morgan}_val.pkl"), 'rb') as f:
                embedding_val_morgan = pickle.load(f)
            with open(os.path.join('prediction','embeddings', f"{save_name_embeddings_morgan}_test.pkl"), 'rb') as f:
                embedding_test_morgan = pickle.load(f)
            embedding_train = {"X": np.concatenate([embedding_train_cat["X"], embedding_train_morgan["X"]], axis=1), "y": embedding_train_cat["y"]}
            embedding_val = {"X": np.concatenate([embedding_val_cat["X"], embedding_val_morgan["X"]], axis=1), "y": embedding_val_cat["y"]}
            embedding_test = {"X": np.concatenate([embedding_test_cat["X"], embedding_test_morgan["X"]], axis=1), "y": embedding_test_cat["y"]}
    else:
        if pred_emb == "CatTransVAE" or pred_emb == "CatTransVAE_emb" or pred_emb == "CatTransVAE_vae":
            # read mols 
            embed_mol_file = pred_data
            df_test = pd.read_csv(os.path.join("prediction", "datasets", embed_mol_file))
            print("Dataset size after sampling:", len(df_test))
            embed_train_mols = df_test[df_test['split']=='train'][_dataset_X].to_numpy()
            embed_train_mols = np.array([Chem.CanonSmiles(s) for s in embed_train_mols])
            embed_train_y = df_test[df_test['split']=='train'][_dataset_y].to_numpy()
            embed_val_mols = df_test[df_test['split']=='val'][_dataset_X].to_numpy()
            embed_val_mols = np.array([Chem.CanonSmiles(s) for s in embed_val_mols])
            embed_val_y = df_test[df_test['split']=='val'][_dataset_y].to_numpy()
            embed_test_mols = df_test[df_test['split']=='test'][_dataset_X].to_numpy()
            embed_test_mols = np.array([Chem.CanonSmiles(s) for s in embed_test_mols])
            embed_test_y = df_test[df_test['split']=='test'][_dataset_y].to_numpy()

            embed_train_mols_idxs = np.arange(len(embed_train_mols))
            train_mems, train_mus, train_logvars, train_embeddings, train_masks = vae.calc_mems(embed_train_mols_idxs, embed_train_mols, return_embedded=True)
            print("MUS shape:", train_mus.shape)
            embed_val_mols_idxs = np.arange(len(embed_val_mols))
            val_mems, val_mus, val_logvars, val_embeddings, val_masks = vae.calc_mems(embed_val_mols_idxs, embed_val_mols, return_embedded=True)
            print("VAL MUS shape:", val_mus.shape)
            embed_test_mols_idxs = np.arange(len(embed_test_mols))
            test_mems, test_mus, test_logvars, test_embeddings, test_masks = vae.calc_mems(embed_test_mols_idxs, embed_test_mols, return_embedded=True)
            print("TEST MUS shape:", test_mus.shape)

            def mean_pooling(embeddings, masks):
                masks = masks.squeeze(1)
                masked_embeddings = embeddings * masks[:, :, None]
                sum_embeddings = masked_embeddings.sum(axis=1)
                count_non_zero = masks.sum(axis=1)[:, None]
                mean_embeddings = sum_embeddings / np.maximum(count_non_zero, 1)
                return mean_embeddings
            
            train_mean_embeddings = mean_pooling(train_embeddings, train_masks)
            print("Train mean embeddings shape:", train_mean_embeddings.shape)
            val_mean_embeddings = mean_pooling(val_embeddings, val_masks)
            print("Val mean embeddings shape:", val_mean_embeddings.shape)
            test_mean_embeddings = mean_pooling(test_embeddings, test_masks)
            print("Test mean embeddings shape:", test_mean_embeddings.shape)

            if pred_emb == "CatTransVAE":
                train_features = np.concatenate([train_mus, train_mean_embeddings], axis=1)
                val_features = np.concatenate([val_mus, val_mean_embeddings], axis=1)
                test_features = np.concatenate([test_mus, test_mean_embeddings], axis=1)
            elif pred_emb == "CatTransVAE_emb":
                train_features = train_mean_embeddings
                val_features = val_mean_embeddings
                test_features = test_mean_embeddings
            elif pred_emb == "CatTransVAE_vae":
                train_features = train_mus
                val_features = val_mus
                test_features = test_mus
            print("Val features shape:", val_features.shape)
            print("Train features shape:", train_features.shape)
            print("Test features shape:", test_features.shape)

            # save embeddings with y to pickle file
            embedding_train = {"X": train_features, "y": embed_train_y}
            embedding_val = {"X": val_features, "y": embed_val_y}
            embedding_test = {"X": test_features, "y": embed_test_y}
            with open(os.path.join('prediction','embeddings', f"{save_name_embeddings}_train.pkl"), 'wb') as f:
                pickle.dump(embedding_train, f)
            with open(os.path.join('prediction','embeddings', f"{save_name_embeddings}_val.pkl"), 'wb') as f:
                pickle.dump(embedding_val, f)
            with open(os.path.join('prediction','embeddings', f"{save_name_embeddings}_test.pkl"), 'wb') as f:
                pickle.dump(embedding_test, f)

        elif pred_emb == "MorganFP":
            def mol_to_fp(mol):
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                arr = np.zeros((1,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(fp, arr)
                return arr

            embed_mol_file = pred_data
            df_test = pd.read_csv(os.path.join("prediction", "datasets", embed_mol_file))
            print("Dataset size after sampling:", len(df_test))
            embed_train_mols = df_test[df_test['split']=='train'][_dataset_X].to_numpy()
            embed_train_y = df_test[df_test['split']=='train'][_dataset_y].to_numpy()
            embed_val_mols = df_test[df_test['split']=='val'][_dataset_X].to_numpy()
            embed_val_y = df_test[df_test['split']=='val'][_dataset_y].to_numpy()
            embed_test_mols = df_test[df_test['split']=='test'][_dataset_X].to_numpy()
            embed_test_y = df_test[df_test['split']=='test'][_dataset_y].to_numpy()

            train_fps = np.array([mol_to_fp(Chem.MolFromSmiles(s)) for s in embed_train_mols])
            val_fps = np.array([mol_to_fp(Chem.MolFromSmiles(s)) for s in embed_val_mols])
            test_fps = np.array([mol_to_fp(Chem.MolFromSmiles(s)) for s in embed_test_mols])

            embedding_train = {"X": train_fps, "y": embed_train_y}
            embedding_val = {"X": val_fps, "y": embed_val_y}
            embedding_test = {"X": test_fps, "y": embed_test_y}

             # save embeddings with y to pickle file
            with open(os.path.join('prediction','embeddings', f"{save_name_embeddings}_train.pkl"), 'wb') as f:
                pickle.dump(embedding_train, f)
            with open(os.path.join('prediction','embeddings', f"{save_name_embeddings}_val.pkl"), 'wb') as f:
                pickle.dump(embedding_val, f)
            with open(os.path.join('prediction','embeddings', f"{save_name_embeddings}_test.pkl"), 'wb') as f:
                pickle.dump(embedding_test, f)
        
    print("#############################")
    print()

    print("Training regression model on embeddings...")

    X_train, y_train = embedding_train["X"], embedding_train["y"]
    X_val, y_val = embedding_val["X"], embedding_val["y"]
    X_test, y_test = embedding_test["X"], embedding_test["y"]
    print("Loaded train embeddings shape:", X_train.shape)
    print("Loaded val embeddings shape:", X_val.shape)
    print("Loaded test embeddings shape:", X_test.shape)

    # Combine train and val sets for 5-fold CV
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    print("Combined train+val embeddings shape:", X_combined.shape)
    print()

    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_combined = scaler.fit_transform(X_combined)
    X_test  = scaler.transform(X_test)

    all_X_combined = np.concatenate([X_combined, X_test], axis=0)
    all_y_combined = np.concatenate([y_combined, y_test], axis=0)

    # Keep SMILES in the same ordering as embeddings: train -> val -> test.
    df_plot = pd.read_csv(os.path.join("prediction", "datasets", pred_data))
    smiles_train = df_plot[df_plot["split"] == "train"][_dataset_X].astype(str).to_numpy()
    smiles_val = df_plot[df_plot["split"] == "val"][_dataset_X].astype(str).to_numpy()
    smiles_test = df_plot[df_plot["split"] == "test"][_dataset_X].astype(str).to_numpy()
    all_smiles_combined = np.concatenate([smiles_train, smiles_val, smiles_test], axis=0)

    # for only suzuki_7054_split_random.csv dataset, group by metal (Ag_0.0)
    if "suzuki_7054" in pred_data:
        group_list = ["group_metal", "group_l1", "group_l2"]
        df_plot["group_metal"] = df_plot["Name"].apply(lambda x: x.split("_")[0])
        df_plot["group_l1"] = df_plot["Name"].apply(lambda x: x.split("_")[1].split(".")[0])
        df_plot["group_l2"] = df_plot["Name"].apply(lambda x: x.split("_")[1].split(".")[1])

        for g in group_list:
            # group = df_plot[g].values
            group_color = g
            group_train = df_plot[df_plot["split"] == "train"][group_color].values
            group_val = df_plot[df_plot["split"] == "val"][group_color].values
            group_test = df_plot[df_plot["split"] == "test"][group_color].values
            group = np.concatenate([group_train, group_val, group_test], axis=0)


            plot_dir = os.path.join('prediction', 'results', save_name)

            plot_outputs = plot_embedding_space_by_value(
                all_X_combined,
                all_y_combined,
                smiles=all_smiles_combined,
                colorby=group,
                save_path=plot_dir,
                file_prefix="embedding_space_" + g,
            )
            print("Saved t-SNE/UMAP plots:", plot_outputs["tsne_path"], plot_outputs["umap_path"])
            if plot_outputs["tsne_html_path"] is not None:
                print("Saved interactive t-SNE HTML:", plot_outputs["tsne_html_path"])

    if "vaskas_1947_all" in pred_data:
        group_list = ["Ligand_A", "Ligand_B"]
        df_plot["Ligand_A"] = df_plot["A"].apply(lambda x: x.split("-")[1])
        df_plot["Ligand_B"] = df_plot["B"].apply(lambda x: x.split("-")[1])
        for g in group_list:
            # group = df_plot[g].values
            group_color = g
            group_train = df_plot[df_plot["split"] == "train"][group_color].values
            group_val = df_plot[df_plot["split"] == "val"][group_color].values
            group_test = df_plot[df_plot["split"] == "test"][group_color].values
            group = np.concatenate([group_train, group_val, group_test], axis=0)


            plot_dir = os.path.join('prediction', 'results', save_name)

            plot_outputs = plot_embedding_space_by_value(
                all_X_combined,
                all_y_combined,
                smiles=all_smiles_combined,
                colorby=group,
                save_path=plot_dir,
                file_prefix="embedding_space_" + g,
            )
            print("Saved t-SNE/UMAP plots:", plot_outputs["tsne_path"], plot_outputs["umap_path"])
            if plot_outputs["tsne_html_path"] is not None:
                print("Saved interactive t-SNE HTML:", plot_outputs["tsne_html_path"])


    plot_dir = os.path.join('prediction', 'results', save_name)

    plot_outputs = plot_embedding_space_by_value(
        all_X_combined,
        all_y_combined,
        smiles=all_smiles_combined,
        colorby=None,
        save_path=plot_dir,
        file_prefix="embedding_space_y",
    )
    print("Saved t-SNE/UMAP plots:", plot_outputs["tsne_path"], plot_outputs["umap_path"])
    if plot_outputs["tsne_html_path"] is not None:
        print("Saved interactive t-SNE HTML:", plot_outputs["tsne_html_path"])

    # dataset = pred_data.split(".")[0]
    # # write results to a text file
    # name = os.path.join('prediction', 'results', f"summary_{args.prediction_dataset}.txt")
    # save_dir = Path("prediction/results") / save_name


    print("#############################")
    print()

if __name__ == '__main__':
    parser = prediction_parser_mol()
    args = parser.parse_args()
    main(args)
