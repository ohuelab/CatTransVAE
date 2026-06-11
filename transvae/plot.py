# plot graphs between training and validation
# of 1) total loss, 2) reconstruction loss, 3) KL loss, 4) Mask loss
# 5) validity, 6) uniqueness 7) novelty, and 8) property prediction loss (if applicable)
# with multiple graph 2 rows and 4 columns, and save the figure as a png file

import matplotlib.pyplot as plt
import seaborn as sns

# Set font to Helvetica
# plt.rcParams["font.sans-serif"] = ["Helvetica"]
plt.rcParams["font.family"] = "sans-serif"

# Alternatively, set directly in sns.set_theme
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(font="Helvetica", style="ticks", rc=custom_params)
sns.set_theme(style="ticks", rc=custom_params)

def plot_total_loss(results, save_path):
    """
    Plots the total loss curves for training and validation.
    
    Args:
        results (pd.DataFrame): A DataFrame containing the training and validation loss history with columns 'epoch', 'train_loss', and 'val_loss'.
        save_path (str): The path to save the plot.
    """
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(results['epoch'], results['train_loss'], label='Train Total Loss')
    plt.plot(results['epoch'], results['val_loss'], label='Validation Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.savefig(save_path)
    # Plot graphs of loss curves
    plt.figure(figsize=(10, 5), dpi=100)
    # log_file_result.write('epoch,train_loss,val_loss,KLBeta,
    # train_recon_loss,train_kl_loss,train_mask_loss,train_prop_loss,
    # val_recon_loss,val_kl_loss,val_mask_loss,val_prop_loss,
    # validity,uniqueness,novelty\n')
    plt.plot(results['epoch'], results['train_loss'], label='Train Loss')
    plt.plot(results['epoch'], results['train_recon_loss'], label='Train Recon Loss')
    plt.plot(results['epoch'], results['train_kl_loss'], label='Train KL Loss')
    if 'train_mask_loss' in results.columns:
        plt.plot(results['epoch'], results['train_mask_loss'], label='Train Mask Loss')
    plt.plot(results['epoch'], results['train_prop_loss'], label='Train Prop Loss')
    plt.plot(results['epoch'], results['val_loss'], label='Val Loss')
    plt.plot(results['epoch'], results['val_recon_loss'], label='Val Recon Loss')
    plt.plot(results['epoch'], results['val_kl_loss'], label='Val KL Loss')
    if 'val_mask_loss' in results.columns:
        plt.plot(results['epoch'], results['val_mask_loss'], label='Val Mask Loss')
    plt.plot(results['epoch'], results['val_prop_loss'], label='Val Prop Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path.replace('.png', '_detail.png'))


def plot_training_history(results, save_path):
    """
    Plots the training and validation history of the model.
    
    Args:
        results (pd.DataFrame): A DataFrame containing the training and validation history with columns for each metric.
        save_path (str): The path to save the plot.
    """
    # Define the metrics to plot
    # log_file_result.write('epoch,train_loss,val_loss,KLBeta,
    # train_recon_loss,train_kl_loss,train_mask_loss,train_prop_loss,
    # val_recon_loss,val_kl_loss,val_mask_loss,val_prop_loss,
    # validity,uniqueness,novelty\n')
    metrics_train = ['train_loss', 'train_recon_loss', 'KLBeta', 'train_kl_loss', 'train_mask_loss', 'train_prop_loss', 'train_reconstruction', 'validity', 'uniqueness', 'novelty']
    metrics_val = ['val_loss', 'val_recon_loss', 'none', 'val_kl_loss', 'val_mask_loss', 'val_prop_loss', 'val_reconstruction', 'none', 'none', 'none']

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 4, figsize=(20, 10), dpi=200)
    
    for i, metric in enumerate(zip(metrics_train, metrics_val)):
        row = i // 4
        col = i % 4
        if metric[0] in results.columns:
            if metric[0] in results.columns:
                axes[row, col].plot(results['epoch'], results[metric[0]], label=metric[0].title())
            if metric[1] in results.columns:
                axes[row, col].plot(results['epoch'], results[metric[1]], label=metric[1].title())
            axes[row, col].set_title(metric[0].replace('_', ' ').title())
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_sample_molecules(smiles_list, save_path, n_samples=30):
    """
    Plots a sample of generated molecules from their SMILES strings.
    
    Args:
        smiles_list (list): A list of SMILES strings representing the molecules to plot.
        save_path (str): The path to save the plot.
        n_samples (int): The number of sample molecules to plot.
    """
    from rdkit import Chem
    from rdkit.Chem import Draw
    
    # Sample a subset of the SMILES list if it's too long
    if len(smiles_list) > n_samples:
        smiles_list = smiles_list[:n_samples]
    
    # Convert SMILES to RDKit molecule objects
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    # Create a grid of molecule images
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200))
    
    # Save the image
    img.save(save_path)

def plot_regression(true_values, predicted_values, measurements, save_path):
    """
    Plots a regression graph comparing true values and predicted values.
    
    Args:
        true_values (list): A list of true values.
        predicted_values (list): A list of predicted values corresponding to the true values.
        measurements (dict): A list of measurement labels for each data point.
    """
    plt.figure(figsize=(6,6), dpi=300)
    plt.scatter(true_values, predicted_values, alpha=0.8, color='#1C3077', edgecolors='none')
    plt.plot([min(0,min(true_values)), max(100,max(true_values))], 
             [min(0,min(true_values)), max(100,max(true_values))], 
             color='grey', linestyle='--')
    plt.xlabel('True Property')
    plt.ylabel('Predicted Property')
    plt.title('Property Prediction')
    # add text of R2 and RMSE
    plt.text(0.05, 0.95, 'MAE: {:.4f}\nRMSE: {:.4f}\nR2: {:.4f}'.format(
        measurements['mae'], measurements['rmse'], measurements['r2']),
        transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.savefig(save_path)
    plt.close()

    plt.figure(figsize=(6,6), dpi=300)
    plt.scatter(true_values, predicted_values, alpha=0.8, color='#1C3077', edgecolors='none')
    range_min_max = max(true_values)
    plt.plot([min(true_values)-0.1*range_min_max, max(true_values)+0.1*range_min_max],
             [min(true_values)-0.1*range_min_max, max(true_values)+0.1*range_min_max],
             color='grey', linestyle='--')
    plt.xlabel('True Property')
    plt.ylabel('Predicted Property')
    plt.title('Property Prediction')
    # add text of R2 and RMSE
    plt.text(0.05, 0.95, 'MAE: {:.4f}\nRMSE: {:.4f}\nR2: {:.4f}'.format(
        measurements['mae'], measurements['rmse'], measurements['r2']),
        transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.savefig(save_path.replace('.png', '_zoom.png'))
    plt.close()
