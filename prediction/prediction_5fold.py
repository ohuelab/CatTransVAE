import os
import sys
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
from catboost import CatBoostRegressor
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

# colors_main = ["#F34D6F"]
colors_main = ["#E37185"]
colors = ["#5971C5", "#F29344", "#458B73", "#F1C54E", "#7955BD", "#5DA6C5", "#96606B", "#ADCA63", "#AE2A2A", "#7D7330"]
light_black = "#595A5A"
black = "#1F1F1F"

def _regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": float(rmse), "MAE": float(mae), "MSE": float(mse), "R2": float(r2)}


def _save_xgboost_feature_importance(model, save_dir, top_k=50):
    importances = np.asarray(model.feature_importances_)
    feature_indices = np.arange(importances.shape[0])

    importance_df = pd.DataFrame(
        {
            "feature_index": feature_indices,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False, kind="mergesort")

    importance_df.to_csv(save_dir / "feature_importance_all.csv", index=False)

    top_df = importance_df.head(min(top_k, len(importance_df))).sort_values(
        "importance", ascending=True, kind="mergesort"
    )

    plt.figure(figsize=(8, max(6, 0.28 * len(top_df))))
    plt.barh(
        top_df["feature_index"].astype(str),
        top_df["importance"],
        color=colors_main[0],
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature Index")
    plt.title(f"XGBoost Feature Importance (Top {len(top_df)})")
    plt.tight_layout()
    plt.savefig(save_dir / "feature_importance_top50.png", dpi=300)
    plt.close()


def _hyperparameter_tuning_5fold(
    X_combined,
    y_combined,
    experiment_name,
    model_type="xgboost",
    n_trials=100,
    n_splits=5,
    random_state=0,
):
    """
    Performs hyperparameter tuning using 5-fold cross-validation on combined train+val set.
    Uses Optuna for xgboost, grid search for randomforest, and simple training for linearregression.
    Returns the best model trained on full combined set.
    """
    save_dir = Path("prediction/results") / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if model_type == "xgboost":
        print(f"Starting XGBoost hyperparameter tuning with 5-fold CV ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                "eta": trial.suggest_float("eta", 0.01, 0.1, log=True),
                "gamma": trial.suggest_float("gamma", 1.0, 8.0),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 8.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 8.0, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, 3000, step=200),
            }
            # params = {
            #     "eta": trial.suggest_float("eta", 0.01, 0.1, log=True),
            #     "max_depth": trial.suggest_int("max_depth", 2, 6),
            #     "min_child_weight": trial.suggest_int("min_child_weight", 1, 3),
            #     "gamma": trial.suggest_float("gamma", 0, 1),
            #     "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            #     "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
            #     "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5, log=True),
            #     "n_estimators": trial.suggest_int("n_estimators", 10, 300, step=10),
            # }
            
            fold_rmses = []
            for train_idx, val_idx in kfold.split(X_combined):
                X_train_fold = X_combined[train_idx]
                y_train_fold = y_combined[train_idx]
                X_val_fold = X_combined[val_idx]
                y_val_fold = y_combined[val_idx]
                
                try:
                    model = XGBRegressor(
                        **params,
                        objective="reg:squarederror",
                        tree_method="hist",
                        random_state=random_state,
                        early_stopping_rounds=50,
                        n_jobs=-1,
                        verbosity=0,
                    )
                    model.fit(
                        X_train_fold,
                        y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        verbose=False,
                    )
                    y_pred_val = model.predict(X_val_fold)
                    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
                    fold_rmses.append(rmse)
                except Exception as e:
                    return float('inf')
            
            return np.mean(fold_rmses)
        
        sampler = TPESampler(seed=random_state)
        pruner = MedianPruner()
        study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_trial.params
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best CV RMSE: {study.best_trial.value:.4f}")
        print(f"Best params: {best_params}")
        
        # Train best model on full combined set
        best_model = XGBRegressor(
            **best_params,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
        best_model.fit(X_combined, y_combined, verbose=False)

        _save_xgboost_feature_importance(best_model, save_dir, top_k=50)
        
        # Save results
        study_df = study.trials_dataframe()
        study_df.to_csv(save_dir / "5fold_optuna_trials.csv", index=False)
        
        with open(save_dir / "best_params_5fold.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)

    elif model_type == "randomforest":
        print(f"Starting RandomForest hyperparameter tuning with 5-fold CV ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }
            
            fold_rmses = []
            for train_idx, val_idx in kfold.split(X_combined):
                X_train_fold = X_combined[train_idx]
                y_train_fold = y_combined[train_idx]
                X_val_fold = X_combined[val_idx]
                y_val_fold = y_combined[val_idx]
                
                try:
                    model = RandomForestRegressor(
                        **params,
                        bootstrap=True,
                        n_jobs=-1,
                        random_state=random_state,
                    )
                    model.fit(X_train_fold, y_train_fold)
                    y_pred_val = model.predict(X_val_fold)
                    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
                    fold_rmses.append(rmse)
                except Exception as e:
                    return float('inf')
            
            return np.mean(fold_rmses)
        
        sampler = TPESampler(seed=random_state)
        pruner = MedianPruner()
        study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_trial.params
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best CV RMSE: {study.best_trial.value:.4f}")
        print(f"Best params: {best_params}")
        
        # Train best model on full combined set
        best_model = RandomForestRegressor(
            **best_params,
            bootstrap=True,
            n_jobs=-1,
            random_state=random_state,
        )
        best_model.fit(X_combined, y_combined)
        
        # Save results
        study_df = study.trials_dataframe()
        study_df.to_csv(save_dir / "5fold_optuna_trials.csv", index=False)
        
        with open(save_dir / "best_params_5fold.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)

    elif model_type == "gradientboosting":
        print(f"Starting GradientBoostingRegressor hyperparameter tuning with 5-fold CV ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }
            
            fold_rmses = []
            for train_idx, val_idx in kfold.split(X_combined):
                X_train_fold = X_combined[train_idx]
                y_train_fold = y_combined[train_idx]
                X_val_fold = X_combined[val_idx]
                y_val_fold = y_combined[val_idx]
                
                try:
                    model = GradientBoostingRegressor(
                        **params,
                        random_state=random_state,
                    )
                    model.fit(X_train_fold, y_train_fold)
                    y_pred_val = model.predict(X_val_fold)
                    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
                    fold_rmses.append(rmse)
                except Exception as e:
                    return float('inf')
            
            return np.mean(fold_rmses)
        
        sampler = TPESampler(seed=random_state)
        pruner = MedianPruner()
        study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_trial.params
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best CV RMSE: {study.best_trial.value:.4f}")
        print(f"Best params: {best_params}")
        
        # Train best model on full combined set
        best_model = GradientBoostingRegressor(
            **best_params,
            random_state=random_state,
        )
        best_model.fit(X_combined, y_combined)
        
        # Save results
        study_df = study.trials_dataframe()
        study_df.to_csv(save_dir / "5fold_optuna_trials.csv", index=False)
        
        with open(save_dir / "best_params_5fold.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)

    elif model_type == "catboost":
        print(f"Starting CatBoostRegressor hyperparameter tuning with 5-fold CV ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 100, 3000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "depth": trial.suggest_int("depth", 3, 15),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 10),
            }
            
            fold_rmses = []
            for train_idx, val_idx in kfold.split(X_combined):
                X_train_fold = X_combined[train_idx]
                y_train_fold = y_combined[train_idx]
                X_val_fold = X_combined[val_idx]
                y_val_fold = y_combined[val_idx]
                
                try:
                    model = CatBoostRegressor(
                        **params,
                        random_state=random_state,
                        verbose=0,
                    )
                    model.fit(X_train_fold, y_train_fold, verbose=False)
                    y_pred_val = model.predict(X_val_fold)
                    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
                    fold_rmses.append(rmse)
                except Exception as e:
                    return float('inf')
            
            return np.mean(fold_rmses)
        
        sampler = TPESampler(seed=random_state)
        pruner = MedianPruner()
        study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_trial.params
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best CV RMSE: {study.best_trial.value:.4f}")
        print(f"Best params: {best_params}")
        
        # Train best model on full combined set
        best_model = CatBoostRegressor(
            **best_params,
            random_state=random_state,
            verbose=0,
        )
        best_model.fit(X_combined, y_combined, verbose=False)
        
        # Save results
        study_df = study.trials_dataframe()
        study_df.to_csv(save_dir / "5fold_optuna_trials.csv", index=False)
        
        with open(save_dir / "best_params_5fold.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2)

    elif model_type == "linearregression":
        print(f"Training LinearRegression with 5-fold CV evaluation...")
        
        # LinearRegression has no hyperparameters to tune, but we evaluate with 5-fold
        fold_rmses = []
        for train_idx, val_idx in kfold.split(X_combined):
            X_train_fold = X_combined[train_idx]
            y_train_fold = y_combined[train_idx]
            X_val_fold = X_combined[val_idx]
            y_val_fold = y_combined[val_idx]
            
            model = LinearRegression(n_jobs=-1)
            model.fit(X_train_fold, y_train_fold)
            y_pred_val = model.predict(X_val_fold)
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
            fold_rmses.append(rmse)
        
        mean_cv_rmse = np.mean(fold_rmses)
        std_cv_rmse = np.std(fold_rmses)
        print(f"\n5-Fold CV RMSE: {mean_cv_rmse:.4f} (+/- {std_cv_rmse:.4f})")
        
        # Train final model on full combined set
        best_model = LinearRegression(n_jobs=-1)
        best_model.fit(X_combined, y_combined)
        
        best_params = {}
        
        # Save CV results
        cv_results = {
            "model": "LinearRegression",
            "mean_cv_rmse": float(mean_cv_rmse),
            "std_cv_rmse": float(std_cv_rmse),
            "fold_rmses": [float(x) for x in fold_rmses],
        }
        with open(save_dir / "5fold_cv_results.json", "w", encoding="utf-8") as f:
            json.dump(cv_results, f, indent=2)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Save best model
    model_path = save_dir / f"{model_type}_best_5fold_model.joblib"
    joblib.dump(best_model, model_path)

    train_val_summary = {
        "experiment_name": experiment_name,
        "model_name": f"{model_type.upper()}_5Fold",
        "best_params": best_params,
    }

    return best_model, best_params, train_val_summary



def test_regressor(model, X_test, y_test, name):
    """
    Evaluates model on test set, saves metrics and scatter plot, returns test summary dict.
    """
    save_dir = Path("prediction/results") / name
    save_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    test_metrics = _regression_metrics(y_test, y_pred)

    # Save metrics
    with open(save_dir / "metrics_test.csv", "w", encoding="utf-8") as f:
        f.write("experiment_name,model,test_RMSE,test_MAE,test_MSE,test_R2\n")
        f.write(
            f"{name},{type(model).__name__},{test_metrics['RMSE']:.4f},{test_metrics['MAE']:.4f},"
            f"{test_metrics['MSE']:.4f},{test_metrics['R2']:.4f}\n"
        )

    # Scatter plot: y_test vs y_pred
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.5, color=colors_main[0])
    min_v = min(np.min(y_test), np.min(y_pred))
    max_v = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v], "--", linewidth=1.5, color="grey")
    plt.text(0.05, 0.95, f"RMSE: {test_metrics['RMSE']:.4f}\nMAE: {test_metrics['MAE']:.4f}\nMSE: {test_metrics['MSE']:.4f}\nR2: {test_metrics['R2']:.4f}", 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Scatter Plot: Actual vs Predicted")
    plt.legend()
    plt.legend().set_visible(False)
    plt.tight_layout()
    plot_path = save_dir / "scatter_test.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return {
        "experiment_name": name,
        "model_name": type(model).__name__,
        "performance": test_metrics,
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
        save_name = f"{pred_data}/{seed}_{pred_emb}_{ckp_model}/{save_name}"
    elif pred_emb == "MorganFP":
        os.makedirs(os.path.join('prediction','results', f"{pred_data}", f"{seed}_{pred_emb}"), exist_ok=True)
        save_name_embeddings = f"{pred_data}/{seed}_{pred_emb}"
        save_name = f"{pred_data}/{seed}_{pred_emb}/{save_name}"
    elif pred_emb == "CatTransVAE+MorganFP":
        os.makedirs(os.path.join('prediction','results', f"{pred_data}", f"{seed}_{pred_emb}_{ckp_model}"), exist_ok=True)
        save_name_embeddings = f"{pred_data}/{seed}_{pred_emb}_{ckp_model}"
        save_name = f"{pred_data}/{seed}_{pred_emb}_{ckp_model}/{save_name}"
    elif pred_emb == "SMI_TED" or pred_emb == "ChemBERTa" or pred_emb == "MoLFormer" or pred_emb == "ChemBERTa_MLM" or pred_emb == "ChemBERTa_MTR" or pred_emb == "MIST":
        os.makedirs(os.path.join('prediction','results', f"{pred_data}", f"{seed}_{pred_emb}"), exist_ok=True)
        save_name_embeddings = f"{pred_data}/{seed}_{pred_emb}"
        save_name = f"{pred_data}/{seed}_{pred_emb}/{save_name}"
    
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
        save_name_embeddings_cat = f"{pred_data}/{seed}_{pred_emb.split('+')[0]}_{ckp_model}"
        save_name_embeddings_morgan = f"{pred_data}/{seed}_{pred_emb.split('+')[1]}"
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

    # Train + validate with 5-fold CV hyperparameter tuning
    model_to_try = args.prediction_model_type
    
    print(f"Starting {model_to_try} with 5-fold cross-validation hyperparameter tuning...")
    best_model, best_params, train_val_summary = _hyperparameter_tuning_5fold(
        X_combined=X_combined,
        y_combined=y_combined,
        experiment_name=save_name,
        model_type=model_to_try,
        n_trials=50,
        n_splits=5,
        random_state=args.seed,
    )
    print(f"\nBest Model Parameters: {best_params}")
    print()

    # Test
    test_summary = test_regressor(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        name=save_name,
    )

    print("Train/Val (5-Fold CV) Summary:")
    print(json.dumps(train_val_summary, indent=2))
    print("\nTest Summary:")
    print(json.dumps(test_summary, indent=2))

    dataset = pred_data.split(".")[0]
    
    # Evaluate on combined set for reference metrics
    y_pred_combined = best_model.predict(X_combined)
    combined_metrics = _regression_metrics(y_combined, y_pred_combined)
    
    # write results to a text file
    if not os.path.exists(os.path.join('prediction', 'results', f"summary_{args.prediction_dataset}.txt")):
        with open(os.path.join('prediction', 'results', f"summary_{args.prediction_dataset}.txt"), 'w', encoding='utf-8') as f:
            f.write("experiment_name,seed,dataset,splitting,embedding,checkpoint_model,model_name,combined_RMSE,combined_MAE,combined_MSE,combined_R2,test_RMSE,test_MAE,test_MSE,test_R2\n")
    with open(os.path.join('prediction', 'results', f"summary_{args.prediction_dataset}.txt"), 'a', encoding='utf-8') as f:
        f.write(f"{save_name},{seed},{dataset},{pred_data},{pred_emb},{ckp_model},{train_val_summary['model_name']},"
                f"{combined_metrics['RMSE']:.4f},{combined_metrics['MAE']:.4f},{combined_metrics['MSE']:.4f},{combined_metrics['R2']:.4f},"
                f"{test_summary['performance']['RMSE']:.4f},{test_summary['performance']['MAE']:.4f},{test_summary['performance']['MSE']:.4f},{test_summary['performance']['R2']:.4f}\n")
    
    # Save detailed summary including best parameters
    save_dir = Path("prediction/results") / save_name
    detailed_summary = {
        "experiment_name": save_name,
        "seed": seed,
        "model_type": model_to_try,
        "best_params": best_params,
        "combined_metrics": combined_metrics,
        "test_metrics": test_summary['performance'],
    }
    with open(save_dir / "summary_detailed_5fold.json", "w", encoding="utf-8") as f:
        json.dump(detailed_summary, f, indent=2)

    # save scaler
    save_dir = Path("prediction/results") / save_name
    with open(save_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("#############################")
    print()

if __name__ == '__main__':
    parser = prediction_parser_mol()
    args = parser.parse_args()
    main(args)
