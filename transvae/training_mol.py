import argparse
from functools import partial
import os
import random
import copy
import numpy as np
from tqdm import tqdm
from time import perf_counter
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.set_deterministic_debug_mode("warn")
# torch.autograd.set_detect_anomaly(True) # make training slower

from transvae.tvae_mol import *
from transvae.tvae_util import *
from transvae.opt import NoamOpt
from transvae.data import vae_data_gen, make_std_mask, MolDataset
from transvae.loss import trans_vae_loss
from transvae.plot import *

####### MODEL SHELL ##########

class VAEShell():
    """
    VAE shell class that includes methods for parameter initiation,
    data loading, training, logging, checkpointing, loading and saving,
    """
    def __init__(self, args, name=None):
        ### Initial
        self.args = args
        self.name = name
        self.seed = self.args.seed if hasattr(self.args, 'seed') else 0
        
        ### Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        print("Using device:", self.device)
        
        ### Functions
        self.loss_func = trans_vae_loss
        self.data_gen = vae_data_gen

        ### Sequence length hard-coded into model
        self.seq_len = 300
        self.src_len = self.seq_len
        self.tgt_len = self.seq_len-1 #300-1 because target is shifted by 1

        ### Build empty structures for data storage
        self.n_epochs = 0
        self.best_loss = np.inf
        self.best_epoch = 0
        self.current_state = {'name': self.name,
                              'epoch': self.n_epochs,
                              'model_state_dict': None,
                              'optimizer_state_dict': None,
                              'property_predictor_external_state_dict': None,
                              'kl_annealer_state_dict': None,
                              'best_loss': self.best_loss,
                              'best_epoch': self.best_epoch,
                              'args': self.args}
        self.loaded_from = None

    def save(self, state, fn, path='checkpoints', use_name=True):
        """
        Saves current model state to .ckpt file

        Arguments:
            state (dict, required): Dictionary containing model state
            fn (str, required): File name to save checkpoint with
            path (str): Folder to store saved checkpoints
        """
        os.makedirs(path, exist_ok=True)
        if use_name:
            if os.path.splitext(fn)[1] == '':
                if self.name is not None:
                    fn += '_' + self.name
                fn += '.ckpt'
            else:
                if self.name is not None:
                    fn, ext = fn.split('.')
                    fn += '_' + self.name
                    fn += '.' + ext
            save_path = os.path.join(path, fn)
        else:
            save_path = fn
        torch.save(state, save_path)

    def train(self, train_idxs, train_mols, val_idxs, val_mols,
              epochs=100, save=True, save_freq=1, log=True, log_dir='trials'):
        """
        Train model and validate

        Arguments:
            train_mols (np.array, required): Numpy array containing training
                                             molecular structures
            val_mols (np.array, required): Same format as train_mols. Used for
                                           model development or validation
                                   molecular structure
            val_props (np.array): Same format as train_prop. Used for model
                                 development or validation
            epochs (int): Number of epochs to train the model for
            save (bool): If true, saves latest and best versions of model
            save_freq (int): Frequency with which to save model checkpoints
            log (bool): If true, writes training metrics to log file
            log_dir (str): Directory to store log files
        """
        ### Prepare data iterators
        print("Preparing train data iterators...")
        train_idx, train_mol = self.data_gen(self.data_source , train_idxs, train_mols, char_dict=self.char_dict)
        print("Preparing val data iterators...")
        val_idx, val_mol = self.data_gen(self.data_source, val_idxs, val_mols, char_dict=self.char_dict)

        print("Building data loaders...")
        train_dataset = MolDataset(train_idx, train_mol)
        val_dataset = MolDataset(val_idx, val_mol)
        train_iter = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=True, num_workers=4,
                                                 pin_memory=torch.cuda.is_available(), drop_last=True)
        val_iter = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=self.batch_size,
                                               shuffle=False, num_workers=4,
                                                pin_memory=torch.cuda.is_available(), drop_last=False)
        self.chunk_size = self.batch_size // self.batch_chunks
        ### Setup log file
        print("Setting up log file...")
        if log:
            log_dir = os.path.join(self.data_dir, self.data_source, log_dir, self.name)
            os.makedirs(log_dir, exist_ok=True)
            if self.name is not None:
                log_fn = '{}/log{}.txt'.format(log_dir, '_'+self.name)
                log_fn_result = '{}/log{}_result.txt'.format(log_dir, '_'+self.name)
            else:
                log_fn = '{}/log.txt'.format(log_dir)
                log_fn_result = '{}/log_result.txt'.format(log_dir)
            try:
                f = open(log_fn, 'r')
                f.close()
                already_wrote = True
                f = open(log_fn_result, 'r')
                f.close()
                already_wrote_result = True
            except FileNotFoundError:
                already_wrote = False
                already_wrote_result = False
            log_file = open(log_fn, 'a')
            log_file_result = open(log_fn_result, 'a')
            if not already_wrote:
                log_file.write('epoch,batch_idx,data_type,tot_loss,recon_loss,pred_loss,kld_loss,prop_mse_loss,run_time\n')
            if not already_wrote_result:
                log_file_result.write('epoch,train_loss,val_loss,KLBeta,train_recon_loss,train_kl_loss,train_mask_loss,train_prop_loss,val_recon_loss,val_kl_loss,val_mask_loss,val_prop_loss,train_reconstruction,val_reconstruction,validity,uniqueness,novelty\n')
            log_file.close()
            log_file_result.close()

        ### Epoch loop
        print("Starting training for {} epochs... which continues from epoch {}".format(epochs, self.n_epochs))
        for e in tqdm(range(epochs)):
            print("Epoch {}/{}...".format(self.n_epochs, epochs))
            ### Train Loop
            time_start = perf_counter()
            self.model.train()
            losses = []
            recon_losses = []
            mask_losses = []
            kl_losses = []
            prop_losses = []
            beta = self.kl_annealer(self.n_epochs)
            for j, data in enumerate(train_iter):
                avg_losses = []
                avg_bce_losses = []
                avg_bcemask_losses = []
                avg_kld_losses = []
                avg_prop_mse_losses = []
                start_run_time = perf_counter()
                data_idx, data_mol = data
                
                for i in range(self.batch_chunks):
                    batch_data_mol = data_mol[i*self.chunk_size:(i+1)*self.chunk_size,:]
                    mols_data = batch_data_mol.to(self.device)
                    print("- Batch {}/{}...{}".format(j*self.batch_chunks+i+1, len(train_iter)*self.batch_chunks, mols_data.shape))

                    src = mols_data.long()
                    tgt = mols_data[:,:-1].long() 
                    src_mask = (src != self.pad_idx).unsqueeze(-2)
                    tgt_mask = make_std_mask(tgt, self.pad_idx)

                    x_out, mu, logvar, pred_len = self.model(src, tgt, src_mask, tgt_mask)

                    assert not torch.isnan(x_out).any(), "TRAIN: x_out contains NaN values."
                    assert not torch.isinf(x_out).any(), "TRAIN: x_out contains Inf values."

                    true_prop = None
                    pred_prop = None
                    true_len = src_mask.sum(dim=-1)
                    loss, bce, bce_mask, kld, prop_mse = self.loss_func(src, x_out, mu, logvar,
                                                                        true_len, pred_len,
                                                                        true_prop, pred_prop,
                                                                        self.char_weights,
                                                                        beta)
                    assert not torch.isnan(loss), "TRAIN: Loss is NaN."
                    assert not torch.isinf(loss), "TRAIN: Loss is Inf."

                    avg_bcemask_losses.append(bce_mask.item())

                    avg_losses.append(loss.item())
                    avg_bce_losses.append(bce.item())
                    avg_kld_losses.append(kld.item())
                    avg_prop_mse_losses.append(prop_mse.item())
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.model.zero_grad()

                stop_run_time = perf_counter()
                run_time = round(stop_run_time - start_run_time, 5)
                avg_loss = np.mean(avg_losses)
                avg_bce = np.mean(avg_bce_losses)
                if len(avg_bcemask_losses) == 0:
                    avg_bcemask = 0
                else:
                    avg_bcemask = np.mean(avg_bcemask_losses)
                avg_kld = np.mean(avg_kld_losses)
                avg_prop_mse = np.mean(avg_prop_mse_losses)
                losses.append(avg_loss)
                recon_losses.append(avg_bce)
                kl_losses.append(avg_kld)
                mask_losses.append(avg_bcemask)
                prop_losses.append(avg_prop_mse)

                if log:
                    log_file = open(log_fn, 'a')
                    log_file.write('{},{},{},{},{},{},{},{},{}\n'.format(self.n_epochs,
                                                                         j, 'train',
                                                                         avg_loss,
                                                                         avg_bce,
                                                                         avg_bcemask,
                                                                         avg_kld,
                                                                         avg_prop_mse,
                                                                         run_time))
                    log_file.close()

            train_loss = np.mean(losses)
            train_recon_loss = np.mean(recon_losses)
            train_kl_loss = np.mean(kl_losses)
            train_mask_loss = np.mean(mask_losses)
            train_prop_loss = np.mean(prop_losses)
            print("[Train time] {} seconds".format(round(perf_counter() - time_start, 5)))

            ### Val Loop
            time_start = perf_counter()
            self.model.eval()
            with torch.no_grad():
                losses = []
                recon_losses = []
                kl_losses = []
                mask_losses = []
                prop_losses = []
                for j, data in enumerate(val_iter):
                    avg_losses = []
                    avg_bce_losses = []
                    avg_bcemask_losses = []
                    avg_kld_losses = []
                    avg_prop_mse_losses = []
                    start_run_time = perf_counter()
                    data_idx, data_mol = data
                    
                    for i in range(self.batch_chunks):
                        batch_data_mol = data_mol[i*self.chunk_size:(i+1)*self.chunk_size,:]

                        if batch_data_mol.size(0) == 0:
                            continue

                        mols_data = batch_data_mol.to(self.device)

                        src = mols_data.long()
                        tgt = mols_data[:,:-1].long()
                        src_mask = (src != self.pad_idx).unsqueeze(-2)
                        tgt_mask = make_std_mask(tgt, self.pad_idx)

                        x_out, mu, logvar, pred_len = self.model(src, tgt, src_mask, tgt_mask)

                        assert not torch.isnan(x_out).any(), "VAL: x_out contains NaN values."
                        assert not torch.isinf(x_out).any(), "VAL: x_out contains Inf values."

                        true_prop = None
                        pred_prop = None
                        true_len = src_mask.sum(dim=-1)
                        loss, bce, bce_mask, kld, prop_mse = self.loss_func(src, x_out, mu, logvar,
                                                                            true_len, pred_len,
                                                                            true_prop, pred_prop,
                                                                            self.char_weights,
                                                                            beta)
                        assert not torch.isnan(loss), "VAL: Loss is NaN."
                        assert not torch.isinf(loss), "VAL: Loss is Inf."

                        avg_bcemask_losses.append(bce_mask.item())

                        avg_losses.append(loss.item())
                        avg_bce_losses.append(bce.item())
                        avg_kld_losses.append(kld.item())
                        avg_prop_mse_losses.append(prop_mse.item())

                    stop_run_time = perf_counter()
                    run_time = round(stop_run_time - start_run_time, 5)
                    avg_loss = np.mean(avg_losses)
                    avg_bce = np.mean(avg_bce_losses)
                    if len(avg_bcemask_losses) == 0:
                        avg_bcemask = 0
                    else:
                        avg_bcemask = np.mean(avg_bcemask_losses)
                    avg_kld = np.mean(avg_kld_losses)
                    avg_prop_mse = np.mean(avg_prop_mse_losses)
                    losses.append(avg_loss)
                    recon_losses.append(avg_bce)
                    mask_losses.append(avg_bcemask)
                    kl_losses.append(avg_kld)
                    prop_losses.append(avg_prop_mse)

                    if log:
                        log_file = open(log_fn, 'a')
                        log_file.write('{},{},{},{},{},{},{},{}\n'.format(self.n_epochs,
                                                                    j, 'val',
                                                                    avg_loss,
                                                                    avg_bce,
                                                                    avg_bcemask,
                                                                    avg_kld,
                                                                    avg_prop_mse,
                                                                    run_time))
                        log_file.close()

            val_loss = np.mean(losses)
            val_recon_loss = np.mean(recon_losses)
            val_kl_loss = np.mean(kl_losses)
            val_mask_loss = np.mean(mask_losses)
            val_prop_loss = np.mean(prop_losses)
            print("[Val time] {} seconds".format(round(perf_counter() - time_start, 5)))          

            time_start = time.perf_counter()
            ### Reconstruction 
            recon_train_idx_subset = train_idxs[:100] if len(train_idxs) >= 100 else train_idxs
            recon_train_mols_subset = train_mols[:100] if len(train_mols) >= 100 else train_mols
            recon_mols = self.reconstruct(recon_train_idx_subset, recon_train_mols_subset)
            matches = sum(1 for original, recon in zip(recon_train_mols_subset, recon_mols) if original == recon)
            recon_percentage_train = matches / len(recon_train_mols_subset) if len(recon_train_mols_subset) > 0 else 0

            recon_val_idx_subset = val_idxs[:100] if len(val_idxs) >= 100 else val_idxs
            recon_val_mols_subset = val_mols[:100] if len(val_mols) >= 100 else val_mols
            recon_mols = self.reconstruct(recon_val_idx_subset, recon_val_mols_subset)
            matches = sum(1 for original, recon in zip(recon_val_mols_subset, recon_mols) if original == recon)
            recon_percentage_val = matches / len(recon_val_mols_subset) if len(recon_val_mols_subset) > 0 else 0
            print("[Recon time] {} seconds".format(round(time.perf_counter() - time_start, 5)))
            
            ### Generate samples
            time_start = time.perf_counter()
            samples = []
            for _ in range(2):
                n_samples_per_batch = 50
                current_samples = self.sample(n_samples_per_batch, sample_mode="rand", sample_dims=None, prompt=None)
                samples.extend(current_samples)
            smiles_gen = samples
            print("[Sample time] {} seconds".format(round(time.perf_counter() - time_start, 5)))

            ### Evaluate samples
            time_start = time.perf_counter()
            smiles_valid = valid(smiles_gen)
            valid_percentage = len(smiles_valid) / len(smiles_gen) if len(smiles_gen) > 0 else 0
            # print("Valid generated:", len(smiles_valid), f"({valid_percentage:.4%})")
            smiles_unique = unique(smiles_valid)
            # unique_percentage = len(smiles_unique) / len(smiles_gen) if len(smiles_gen) > 0 else 0
            unique_percentage_valid = len(smiles_unique) / len(smiles_valid) if len(smiles_valid) > 0 else 0
            # print("Valid&Unique generated:", len(smiles_unique), f"({unique_percentage:.4%})", f"({unique_percentage_valid:.4%} of valid)")
            
            if self.n_epochs % 10 == 0 or self.n_epochs == epochs-1:
                smiles_train = train_mols.tolist()
                smiles_novel = novel(smiles_unique, smiles_train)
                # novel_percentage = len(smiles_novel) / len(smiles_gen) if len(smiles_gen) > 0 else 0
                novel_percentage_validunique = len(smiles_novel) / len(smiles_unique) if len(smiles_unique) > 0 else 0
                # print("Valid&Unique&Novel generated:", len(smiles_novel), f"({novel_percentage:.4%})", f"({novel_percentage_validunique:.4%} of valid&unique)")
            else:
                novel_percentage_validunique = -0.1
            print("[Metric time] {} seconds".format(round(time.perf_counter() - time_start, 5)))

            print('==== Epoch: {}\tTrain: {}\tVal: {}\tKLBeta: {}\t' \
            'T_Recon: {}\tT_KL: {}\tT_Mask: {}\tT_Prop: {}\t' \
            'V_Recon: {}\tV_KL: {}\tV_Mask: {}\tV_Prop: {}'.format(self.n_epochs, train_loss, val_loss, beta, 
                                                       train_recon_loss, train_kl_loss, train_mask_loss, train_prop_loss, 
                                                       val_recon_loss, val_kl_loss, val_mask_loss, val_prop_loss))

            if log:
                log_file_result = open(log_fn_result, 'a')
                log_file_result.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(self.n_epochs,
                                                            train_loss,
                                                            val_loss,
                                                            beta,
                                                            train_recon_loss,train_kl_loss,train_mask_loss,train_prop_loss,
                                                            val_recon_loss,val_kl_loss,val_mask_loss,val_prop_loss,
                                                            recon_percentage_train, recon_percentage_val, 
                                                            valid_percentage, unique_percentage_valid, novel_percentage_validunique))
                log_file_result.close()
                # Plot graphs of loss curves
                results = pd.read_csv(log_fn_result)
                plot_total_loss(results, save_path=log_fn_result.replace('.txt', '.png'))
                plot_training_history(results, save_path=log_fn_result.replace('.txt', '_all.png'))
                if len(smiles_valid) > 0:
                    plot_sample_molecules(smiles_valid, save_path=log_fn_result.replace('.txt', '_samples_{}.png'.format(self.n_epochs)), n_samples=30)
            
            ### Update current state
            self.current_state['epoch'] = self.n_epochs
            self.current_state['model_state_dict'] = self.model.state_dict()
            self.current_state['optimizer_state_dict'] = self.optimizer.state_dict()
            self.current_state['kl_annealer_state_dict'] = self.kl_annealer.state_dict()

            save_start = self.save_start if hasattr(self, 'save_start') else 0
            if val_loss < self.best_loss and self.n_epochs > save_start:
            # if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.current_state['best_loss'] = self.best_loss
                self.best_epoch = self.n_epochs
                self.current_state['best_epoch'] = self.best_epoch
                if save:
                    self.save(self.current_state, 'best', path=os.path.join(self.data_dir, self.data_source, 'checkpoints'))

            if (self.n_epochs) % save_freq == 0 and self.n_epochs >= save_start:
                epoch_str = str(self.n_epochs)
                while len(epoch_str) < 3:
                    epoch_str = '0' + epoch_str
                if save:
                    self.save(self.current_state, epoch_str, path=os.path.join(self.data_dir, self.data_source, 'checkpoints'))
            
            ### Add epoch
            self.n_epochs += 1
            print()

            
    ### Placeholder functions to be implemented in child class
    def test(self, test_idx, test_mols):
        pass

    ### Sampling and Decoding Functions
    def sample_from_memory(self, size, mode='rand', sample_dims=None, k_entropy=5):
        """
        Quickly sample from latent dimension

        Arguments:
            size (int, req): Number of samples to generate in one batch
            mode (str): Sampling mode (rand, high_entropy or k_high_entropy, rand_training, rand_target)
            sample_dims (list): List of dimensions to sample 
            k_entropy (int): Number of high entropy dimensions to randomly sample from
        Returns:
            z (torch.tensor): NxD_latent tensor containing sampled memory vectors
        """
        if mode == 'rand':
            z = torch.randn(size, self.d_latent)
        elif mode == 'rand_training':
            low = torch.tensor([d[0] for d in sample_dims], dtype=torch.float)
            high = torch.tensor([d[1] for d in sample_dims], dtype=torch.float)
            z = low + (high - low) * torch.rand(size, self.d_latent)
        elif mode == 'rand_target':
            sample_dims_list = [torch.tensor(d, dtype=torch.float) for d in sample_dims]
            z = random.choices(sample_dims_list, k=size)[0].unsqueeze(0)
            noise = torch.normal(mean=0, std=1, size=(size, self.d_latent)) * 0.5
            z = z + noise
        elif mode == 'high_entropy':
            z = torch.zeros((size, self.d_latent))
            for d in sample_dims:
                z[:,d] = torch.randn(size)
        elif mode == 'k_high_entropy':
            z = torch.zeros((size, self.d_latent))
            d_select = np.random.choice(sample_dims, size=k_entropy, replace=False)
            for d in d_select:
                z[:,d] = torch.randn(size)
        return z
    
    def decode_sample(self, mem, mem_pad_mask, target, target_pad_mask, temperature=1.0, top_k=None):
        """
        Decode a sample from the latent space

        Arguments:
            mem (torch.tensor, req): NxD_latent tensor containing sampled memory vectors
            mem_pad_mask (torch.tensor, req): Mask for memory padding
            target (torch.tensor, req): Target tensor to decode
            target_pad_mask (torch.tensor, req): Mask for target padding
        """
        output = self.model.decode(mem, mem_pad_mask, target, target_pad_mask)
        output = self.model.generator(output)
        logits = output[:,-1,:]
        # temperature
        logits = logits / (temperature + 1e-8)
        # top_k (optional)
        if top_k is not None:
            if int(top_k) > 0:
                top_k = min(top_k, logits.size(-1))
                top_logits, top_indices = torch.topk(logits, top_k)
                logits[logits < top_logits[:, -1].unsqueeze(1)] = -float('Inf')

        # ---- ADD DICTIONARY SYNTAX CONSTRAINT RULE ----
        last_tokens = target[:, -1]-1  # (B,)
        if all(last_tokens == -1):
            mask_disallow = self.dict_mask_disallow_start.expand(logits.size(0), -1)  # (B, vocab_size-1)
        else:
            mask_disallow = self.dict_mask_disallow[last_tokens]  # (B, vocab_size-1)
        logits = logits.masked_fill(mask_disallow, -float('Inf'))
        # -----------------------------------------------

        prob = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return output, logits, prob, log_probs

    def greedy_decode(self, mem, src_mask=None, condition=[], limit_max_len=300, temperature=1.0, top_k=None, do_sample=False):
        """
        Greedy decode from model memory

        Arguments:
            mem (torch.tensor, req): Memory tensor to send to decoder
            src_mask (torch.tensor): Mask tensor to hide padding tokens 
        Returns:
            decoded (torch.tensor): Tensor of predicted token ids
        """
        start_symbol = self.start_symbol
        max_len = min(limit_max_len, self.tgt_len)
        decoded = torch.ones(mem.shape[0],1).fill_(start_symbol).long().to(self.device)
        condition = condition[:self.tgt_len-2] # ensure condition is not longer than max target length
        condition_len = len(condition)
        for tok in condition:
            try:
                condition_symbol = self.char_dict[tok]
            except KeyError:
                print(f"Warning: condition token '{tok}' not found in char_dict. Skipping this token in decoding.")
                print(condition)
                condition_len -= 1
                continue
            condition_vec = torch.ones(mem.shape[0],1).fill_(condition_symbol).long().to(self.device)
            decoded = torch.cat([decoded, condition_vec], dim=1)
        tgt = torch.ones(mem.shape[0],max_len+1).fill_(start_symbol).long().to(self.device)
        tgt[:,:condition_len+1] = decoded

        if src_mask is None:
            mask_lens = self.model.encoder.predict_mask_length(mem)
            src_mask = torch.zeros((mem.shape[0], 1, self.src_len)).to(self.device)
            for i in range(mask_lens.shape[0]):
                mask_len = mask_lens[i].item()
                src_mask[i,:,:mask_len] = torch.ones((1, 1, mask_len)).to(self.device)

        self.model.eval()
        g = torch.Generator(device=self.device).manual_seed(self.seed)
        with torch.no_grad():
            for i in range(condition_len, max_len):
                decode_mask = subsequent_mask(decoded.size(1)).long().to(self.device)
                decoded = decoded.to(self.device)

                output, logits, prob, log_probs = self.decode_sample(mem, 
                                                                    src_mask, 
                                                                    decoded, 
                                                                    decode_mask, 
                                                                    temperature=temperature, 
                                                                    top_k=top_k)
                # sample
                if do_sample:
                    next_word = torch.multinomial(prob, num_samples=1, generator=g).squeeze(1)
                else:
                    _, next_word = torch.max(prob, dim=1)


                # Generate next elements in the pad mask. An element is padded if:
                # 1. The previous token is an end token
                # 2. The previous token is a pad token
                if i > condition_len:
                    is_end_token = decoded[:, i - 1] == self.end_symbol
                    is_pad_token = decoded[:, i - 1] == self.pad_idx
                    new_pad_mask = torch.logical_or(is_end_token, is_pad_token)

                    # Break if sampling is complete
                    if new_pad_mask.sum().item() == new_pad_mask.numel():
                        break

                    # Ensure all sequences contain an end token
                    if i == max_len-1:
                        next_word[~new_pad_mask] = self.end_symbol-1

                    # Set the token to pad where required
                    next_word[new_pad_mask] = self.pad_idx-1
                
                next_word += 1
                tgt[:,i+1] = next_word

                next_word = next_word.unsqueeze(1)
                decoded = torch.cat([decoded, next_word], dim=1)
            
        decoded = tgt[:,1:]
        return decoded
    
    def beam_decode(self, mem, src_mask=None, condition=[], beam_width=10, limit_max_len=300):
        from transvae.beam import _beam_step, _update_beams_, _transpose_list, _sort_beams
        """
        Beam search decode from model memory

        Arguments:
            mem (torch.tensor, req): Memory tensor to send to decoder
            src_mask (torch.tensor): Mask tensor to hide padding tokens (if
                                     model_type == 'transformer')
        """
        start_symbol = self.start_symbol
        pad_idx = self.pad_idx
        end_symbol = self.end_symbol
        max_len = min(self.tgt_len+1, limit_max_len)
        batch_size = self.batch_size if self.batch_size < mem.shape[0] else mem.shape[0]
        mem = mem.to(self.device)
        
        # Create tensors which will be reused
        token_ids = [start_symbol] + ([pad_idx] * (max_len - 1))
        token_ids = [token_ids] * batch_size
        token_ids = torch.tensor(token_ids, device=self.device)
        for tok_i, tok in enumerate(condition):
            condition_symbol = self.char_dict[tok]
            condition_vec = torch.ones(mem.shape[0],1).fill_(condition_symbol).long().to(self.device)
            # token_ids = torch.cat([token_ids, condition_vec], dim=1)
            token_ids[:,tok_i+1] = condition_vec
        token_ids = token_ids.transpose(0, 1).to(self.device)
        pad_mask = torch.zeros((max_len, batch_size), device=self.device, dtype=torch.bool)

        ts = token_ids[:len(condition)+1, :]
        ms = pad_mask[:len(condition)+1, :]
        ll = torch.zeros((batch_size), device=self.device)

        # decode function
        if src_mask is None:
            mask_lens = self.model.encoder.predict_mask_length(mem)
            src_mask = torch.zeros((mem.shape[0], 1, self.src_len)).to(self.device)
            for i in range(mask_lens.shape[0]):
                mask_len = mask_lens[i].item()
                src_mask[i,:,:mask_len] = torch.ones((1, 1, mask_len)).to(self.device)

        self.model.eval()
        decode_fn = partial(self.decode_sample, mem=mem, mem_pad_mask=src_mask)

        k = beam_width
        with torch.no_grad():
            ms = subsequent_mask(ts.size(0)).long().to(self.device)
            # Apply starting token to model to get a distribution over next tokens
            first_lls = _beam_step(decode_fn, ts, ms, ll, pad_token_id=pad_idx, end_token_id=end_symbol)
            top_lls, top_idxs = torch.topk(first_lls, k, dim=1)
            top_ids = list(top_idxs.T)
            top_ids = [ids+1 for ids in top_ids]

            # Setup tensors for each beam which will be reused
            token_ids_list = [token_ids.clone() for _ in range(k)]
            pad_mask_list = [pad_mask.clone() for _ in range(k)]
            lls_list = list(top_lls.T)

            for beam_idx, ids in enumerate(top_ids):
                token_ids_list[beam_idx][len(condition)+1, :] = ids
                pad_mask_list[beam_idx][len(condition)+1, :] = 0

            for i in range(len(condition)+2, max_len):
                complete = _update_beams_(i, decode_fn, token_ids_list, pad_mask_list, lls_list, 
                                          pad_token_id=pad_idx, end_token_id=end_symbol, max_seq_len=max_len)
                if complete:
                    break

            tokens_list = [token_ids.transpose(0, 1) for token_ids in token_ids_list]

            mol_strs_list = [tokens for tokens in tokens_list]
            log_lhs_list = [log_lhs.tolist() for log_lhs in lls_list]

            # Transpose and sort list of molecules based on ll
            new_mol_strs = _transpose_list(mol_strs_list)
            new_log_lhs = _transpose_list(log_lhs_list)
            sorted_mols, sorted_lls = _sort_beams(new_mol_strs, new_log_lhs)

        # return sorted_mols, sorted_lls
        return sorted_mols

    def reconstruct(self, data_idxs, data_mols, method='greedy', return_mems=False, return_str=True):
        """
        Method for encoding input smiles into memory and decoding back into smiles

        Arguments:
            data (np.array, required): Input array consisting of smiles and property
            method (str): Method for decoding. Greedy decoding is currently the only
                          method implemented. May implement beam search, top_p or top_k
                          in future versions.
            log (bool): If true, tracks reconstruction progress in separate log file
            return_mems (bool): If true, returns memory vectors in addition to decoded SMILES
            return_str (bool): If true, translates decoded vectors into SMILES strings. If false
                               returns tensor of token ids
        Returns:
            decoded_smiles (list): Decoded smiles data - either decoded SMILES strings or tensor of
                                   token ids
            mems (np.array): Array of model memory vectors
        """
        batch_size_recon = self.batch_size if self.batch_size < len(data_mols) else len(data_mols)
        batch_chunks_recon = self.batch_chunks if self.batch_chunks < batch_size_recon else 1
        data_idx, data_mol = self.data_gen(self.data_source, data_idxs, data_mols, char_dict=self.char_dict)
        
        data_dataset = MolDataset(data_idx, data_mol)
        
        ### Prepare data iterator
        data_iter = torch.utils.data.DataLoader(data_dataset,
                                                batch_size=batch_size_recon,
                                                shuffle=False, num_workers=4,
                                                pin_memory=False, drop_last=False)
        save_shape = len(data_mols)
        chunk_size_recon = batch_size_recon // batch_chunks_recon
        mems = torch.empty((save_shape, self.d_latent)).to(self.device)

        self.model.eval()
        with torch.no_grad():
            decoded_smiles = []
            for j, data in tqdm(enumerate(data_iter), total=len(data_iter), desc="Reconstructing SMILES"):
                data_idx, data_mol = data
                for i in range(batch_chunks_recon):
                    batch_data_mol = data_mol[i*chunk_size_recon:(i+1)*chunk_size_recon,:]
                    mols_data = batch_data_mol.to(self.device)

                    src = mols_data.long()
                    src_mask = (src != self.pad_idx).unsqueeze(-2)

                    ### Run through encoder to get memory (use mu as mem)
                    _, mem, _, _ = self.model.encode(src, src_mask)

                    start = j*batch_size_recon+i*chunk_size_recon
                    stop = j*batch_size_recon+(i+1)*chunk_size_recon if (j*batch_size_recon+(i+1)*chunk_size_recon) < data_mol.shape[0] else j*batch_size_recon+data_mol.shape[0]
                    mems[start:stop, :] = mem.detach().cpu()

                    ### Decode logic
                    if method == 'greedy':
                        decoded = self.greedy_decode(mem, src_mask=src_mask)
                    elif method == 'beam':
                        decoded = self.beam_decode(mem, src_mask=src_mask, beam_width=10)
                    else:
                        decoded = None

                    if return_str:
                        if method == 'greedy':
                            decoded = decode_mols(decoded, self.org_dict)
                            decoded_smiles += decoded
                        elif method == 'beam':
                            for d in decoded:
                                decoded_smiles += [decode_mols(torch.tensor(tokens[1:]).unsqueeze(0), self.org_dict) for tokens in d][0]
                    else:
                        decoded_smiles.append(decoded)

        if return_mems:
            return decoded_smiles, mems.detach().numpy()
        else:
            return decoded_smiles

    def evaluate(self, data_mols):
        """
        Method for evaluating the reconstruction performance of the model

        Arguments:
            smiles (list, req): List of original SMILES strings
        """
        condition = [tokenizer(smiles) for smiles in data_mols]
        # limit the length
        condition = [c[:self.tgt_len-2] if len(c) > self.tgt_len-2 else c for c in condition]
        data_idxs = list(range(len(data_mols)))

        batch_size_eval = 1
        batch_chunks_eval = 1
        chunk_size_eval = 1 #self.batch_size // self.batch_chunks

        data_idx, data_mol = self.data_gen(self.data_source, data_idxs, data_mols, char_dict=self.char_dict)
        data_dataset = MolDataset(data_idx, data_mol)
        ### Prepare data iterator
        data_iter = torch.utils.data.DataLoader(data_dataset,
                                                batch_size=batch_size_eval,
                                                shuffle=False, num_workers=4,
                                                pin_memory=False, drop_last=False)

        save_shape = len(data_iter)*batch_size_eval
        mems = torch.empty((save_shape, self.d_latent)).cpu()

        self.model.eval()
        with torch.no_grad():
            nlls = []
            for j, data in tqdm(enumerate(data_iter), total=len(data_iter), desc="Evaluate SMILES"):
                data_idx, data_mol = data
                for i in range(batch_chunks_eval):
                    batch_data_mol = data_mol[i*chunk_size_eval:(i+1)*chunk_size_eval,:]
                    mols_data = batch_data_mol.to(self.device)

                    src = mols_data.long()
                    src_mask = (src != self.pad_idx).unsqueeze(-2)

                    ### Run through encoder to get memory
                    _, mem, _, _ = self.model.encode(src, src_mask)

                    start = j*batch_size_eval+i*chunk_size_eval
                    stop = j*batch_size_eval+(i+1)*chunk_size_eval
                    mems[start:stop, :] = mem.detach().cpu()

                    start_symbol = self.start_symbol
                    max_len = self.tgt_len
                    decoded = torch.ones(mem.shape[0],1).fill_(start_symbol).long().to(self.device)
                    
                    condition_j = condition[j]
                    condition_j = condition_j[:self.tgt_len-2] # ensure condition is not longer than max target length
                    condition_j_len = len(condition_j)
                    for tok in condition_j:
                        try:
                            condition_symbol = self.char_dict[tok]
                        except KeyError:
                            print(f"Warning: condition token '{tok}' not found in char_dict. Skipping this token in decoding.")
                            print(condition)
                            condition_j_len -= 1
                            continue
                        condition_vec = torch.ones(mem.shape[0],1).fill_(condition_symbol).long().to(self.device)
                        decoded = torch.cat([decoded, condition_vec], dim=1)
                    tgt = torch.ones(mem.shape[0],max_len+1).fill_(start_symbol).long().to(self.device)
                    tgt[:,:condition_j_len+1] = decoded
                    
                    if src_mask is None:
                        mask_lens = self.model.encoder.predict_mask_length(mem)
                        src_mask = torch.zeros((mem.shape[0], 1, self.src_len)).to(self.device)
                        for i in range(mask_lens.shape[0]):
                            mask_len = mask_lens[i].item()
                            src_mask[i,:,:mask_len] = torch.ones((1, 1, mask_len)).to(self.device)

                    for c in range(condition_j_len, condition_j_len+1):
                        decode_mask = subsequent_mask(decoded.size(1)).long().to(self.device)
                        decoded = decoded.to(self.device)

                        output, logits, prob, log_probs = self.decode_sample(mem, 
                                                            src_mask, 
                                                            decoded, 
                                                            decode_mask, 
                                                            temperature=1.0, 
                                                            top_k=None)
                        nll_loss = F.nll_loss(log_probs, tgt[:,c], reduction='none')
                        nlls.append(nll_loss.item())
        return nlls

    def sample(self, n, method='greedy', sample_mode='rand',
                        sample_dims=None, k_entropy=None, return_str=True,
                        prompt=None, batch_size=1, limit_max_len=300, temperature=1.0, top_k=None, do_sample=False):
        """
        Method for sampling from memory and decoding back into SMILES strings

        Arguments:
            n (int): Number of data points to sample
            method (str): Method for decoding. Greedy decoding is currently the only
                          method implemented. May implement beam search, top_p or top_k
                          in future versions.
            sample_mode (str): Sampling mode (rand, high_entropy or k_high_entropy)
            sample_dims (list): List of dimensions to sample from if mode is
                                high_entropy or k_high_entropy
            k_entropy (int): Number of high entropy dimensions to randomly sample from
            return_str (bool): If true, translates decoded vectors into SMILES strings. If false
                               returns tensor of token ids
        Returns:
            decoded (list): Decoded smiles data - either decoded SMILES strings or tensor of
                            token ids
        """
        # tokenize prompt
        condition = tokenizer(prompt) if prompt is not None else []

        mem = self.sample_from_memory(n, mode=sample_mode, sample_dims=sample_dims, k_entropy=k_entropy).to(self.device)

        ### Decode logic
        if method == 'greedy':
            decoded = self.greedy_decode(mem, condition=condition, 
                                         limit_max_len=limit_max_len, 
                                         temperature=temperature, 
                                         top_k=top_k, 
                                         do_sample=do_sample)
        elif method == 'beam':
            decoded = self.beam_decode(mem, condition=condition, 
                                       beam_width=10, limit_max_len=limit_max_len)
        else:
            decoded = None

        if return_str:
            if method == 'greedy':
                decoded = decode_mols(decoded, self.org_dict)
            elif method == 'beam':
                decoded_all = []
                for d in decoded:
                    decoded_all.extend([decode_mols(torch.tensor(tokens[1:]).unsqueeze(0), self.org_dict)[0] for tokens in d][0:1])
                decoded = decoded_all

        return decoded

    def calc_mems(self, data_idxs, data_mols, save_dir='memory', save_fn='model_name', return_embedded=False, save=False):
        """
        Method for calculating and saving the memory of each neural net

        Arguments:
            data (np.array, req): Input array containing SMILES strings
            log (bool): If true, tracks calculation progress in separate log file
            save_dir (str): Directory to store output memory array
            save_fn (str): File name to store output memory array
            return_embedded (bool): If true, returns the embedded representations along with the memory
            save (bool): If true, saves memory to disk. If false, returns memory
        Returns:
            mems(np.array): Reparameterized memory array
            mus(np.array): Mean memory array (prior to reparameterization)
            logvars(np.array): Log variance array (prior to reparameterization)
        """
        if save_fn == 'model_name':
            save_fn = self.name
        if save:
            save_dir = os.path.join(self.data_dir, self.data_source, f"mems")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, save_fn)
            if os.path.exists('{}_mems.npy'.format(save_path)):
                print(f"Memory file {save_path}_mems.npy already exists. Loading from file.")
                mems = np.load('{}_mems.npy'.format(save_path))
                mus = np.load('{}_mus.npy'.format(save_path))
                logvars = np.load('{}_logvars.npy'.format(save_path))
                return mems, mus, logvars

        batch_size_cals = self.batch_size if self.batch_size < len(data_mols) else len(data_mols)
        batch_chunks_cals = self.batch_chunks if self.batch_chunks < batch_size_cals else 1
        data_idx, data_mol = self.data_gen(self.data_source, data_idxs, data_mols, char_dict=self.char_dict)
        
        data_dataset = MolDataset(data_idx, data_mol)

        data_iter = torch.utils.data.DataLoader(data_dataset,
                                                batch_size=batch_size_cals,
                                                shuffle=False, num_workers=4,
                                                pin_memory=False, drop_last=False)
        
        save_shape = len(data_mol)
        chunk_size_cals = batch_size_cals // batch_chunks_cals
        mems = torch.empty((save_shape, self.d_latent)).cpu()
        mus = torch.empty((save_shape, self.d_latent)).cpu()
        logvars = torch.empty((save_shape, self.d_latent)).cpu()
        embeddings = torch.empty((save_shape, self.seq_len, self.d_model)).cpu() if return_embedded else None
        masks = torch.empty((save_shape, 1, self.seq_len)).cpu() if return_embedded else None

        self.model.eval()
        with torch.no_grad():
            for j, data in tqdm(enumerate(data_iter), total=len(data_iter), desc='Calculating mems'):
                data_idx, data_mol = data
                for i in range(batch_chunks_cals):
                    batch_data_mol = data_mol[i*chunk_size_cals:(i+1)*chunk_size_cals,:]
                    mols_data = batch_data_mol.to(self.device)

                    src = mols_data.long()
                    tgt = mols_data[:,:-1].long()
                    src_mask = (src != self.pad_idx).unsqueeze(-2)
                    tgt_mask = make_std_mask(tgt, self.pad_idx)

                    ### Run through encoder to get memory
                    if return_embedded:
                        mem, mu, logvar, _, embedding = self.model.encode(src, src_mask, return_embedded=True)
                    else:
                        mem, mu, logvar, _ = self.model.encode(src, src_mask)
                    
                    start = j*batch_size_cals+i*chunk_size_cals
                    stop = j*batch_size_cals+(i+1)*chunk_size_cals if (j*batch_size_cals+(i+1)*chunk_size_cals) < data_mol.shape[0] else j*batch_size_cals+data_mol.shape[0]
                    mems[start:stop, :] = mem.detach().cpu()
                    mus[start:stop, :] = mu.detach().cpu()
                    logvars[start:stop, :] = logvar.detach().cpu()
                    if return_embedded:
                        embeddings[start:stop, :, :] = embedding.detach().cpu()
                        masks[start:stop, :, :] = src_mask.detach().cpu()
            if save:
                if save_fn == 'model_name':
                    save_fn = self.name
                save_dir = os.path.join(self.data_dir, self.data_source, f"mems")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, save_fn)
                np.save('{}_mems.npy'.format(save_path), mems.detach().numpy())
                np.save('{}_mus.npy'.format(save_path), mus.detach().numpy())
                np.save('{}_logvars.npy'.format(save_path), logvars.detach().numpy())
                if return_embedded: 
                    np.save('{}_embeddings.npy'.format(save_path), embeddings.detach().numpy())
                    np.save('{}_masks.npy'.format(save_path), masks.detach().numpy())
            if return_embedded:
                return mems.detach().numpy(), mus.detach().numpy(), logvars.detach().numpy(), embeddings.detach().numpy(), masks.detach().numpy()
            return mems.detach().numpy(), mus.detach().numpy(), logvars.detach().numpy()


class TransVAE(VAEShell):
    """
    Transformer-based VAE class. Between the encoder and decoder is a stochastic
    latent space. "Memory value" matrices are convolved to latent bottleneck and
    deconvolved before being sent to source attention in decoder.
    """
    def __init__(self, args={}, name=None, mode="training", load_fn=None, finetune=False):
        super().__init__(args, name)
        """
        Instatiating a TransVAE object builds the model architecture, data structs
        to store the model parameters and training information and initiates model
        weights. Most params have default options but vocabulary must be provided.

        Arguments:
            args (dict, required): Dictionary with model arguments. Keys must match
                                     those written in this module
            name (str): Name of model (all save and log files will be written with
                        this name)
            load_fn (str): Path to checkpoint file
        """
        ### Store architecture params
        self.args = args if args != {} else argparse.Namespace()
        self.seed = args.seed if hasattr(args, 'seed') else 0
        self.model_type = args.model_type if hasattr(args, 'model_type') else 'transvae'
        self.data_source = args.data_source if hasattr(args, 'data_source') else 'pubchem10M'
        self.data_dir = args.data_dir if hasattr(args, 'data_dir') else 'data'
        
        self.arch_params = ['N', 'd_model', 'd_ff', 'd_latent', 'h', 'dropout', 'bypass_bottleneck', 
                            'eps_scale']
        self.args_inputs = ['data_source', 'epochs', 'batch_size', 'batch_chunks', 'adam_lr', 
                              'kl_n_epoch', 'anneal_start', 'beta', 'beta_init', 'lr_scale', 
                              'warmup_steps', 'save_start', 'expansion', 'augmentation', 'checkpoint',
                              'finetune', 'save_freq', 'seed', 'data_dir', 'save_name']
        self.args_form_pretrained = ['vocab_path', 'char_weights_path', 'char_dict', 'char_weights']
        ### Build model architecture
        if load_fn is None:
            self.build_model(mode=mode)
        else:
            self.load(checkpoint_path=load_fn, mode=mode, finetune=finetune)

        # print("Current Args:", self.args)
        print("✓ Model architecture built successfully.")


    def build_model(self, mode="training"):
        """
        Build model architecture. This function is called during initialization as well as when
        loading a saved model checkpoint
        """
        self.N = self.args.N if hasattr(self.args, 'N') else 3
        self.d_model = self.args.d_model 
        self.d_ff = self.args.d_ff if hasattr(self.args, 'd_ff') else 512
        self.d_latent = self.args.d_latent
        self.h = self.args.h if hasattr(self.args, 'h') else 8
        self.dropout = self.args.dropout if hasattr(self.args, 'dropout') else 0.1
        self.bypass_bottleneck = self.args.bypass_bottleneck if hasattr(self.args, 'bypass_bottleneck') else False
        self.eps_scale = self.args.eps_scale
        self.batch_size = self.args.batch_size
        self.batch_chunks = self.args.batch_chunks

        if mode == "training":
            # Training params
            self.adam_lr = self.args.adam_lr
            self.kl_n_epoch = self.args.kl_n_epoch
            self.anneal_start = self.args.anneal_start
            self.beta = self.args.beta
            self.beta_init = self.args.beta_init
            self.lr_scale = self.args.lr_scale
            self.warmup_steps = self.args.warmup_steps
            self.save_start = self.args.save_start

        self.char_dict = self.args.char_dict
        self.char_weights = self.args.char_weights
        self.org_dict = self.args.org_dict
        self.vocab_size = len(self.char_dict.keys())
        self.pad_idx = self.char_dict['_']
        self.start_symbol = self.char_dict['<start>']
        self.end_symbol = self.char_dict['<end>']
        self.unk_symbol = self.char_dict['<unk>']
        
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.h, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        # Embedding
        src_embed = nn.Sequential(Embeddings(self.d_model, self.vocab_size), c(position))
        tgt_embed = nn.Sequential(Embeddings(self.d_model, self.vocab_size), c(position))
        # TransVAE
        encoder = VAEEncoder(EncoderLayer(self.d_model, self.src_len, c(attn), c(ff), self.dropout),
                                          self.N, self.d_latent, self.src_len, self.bypass_bottleneck,
                                          self.eps_scale)
        decoder = VAEDecoder(EncoderLayer(self.d_model, self.src_len, c(attn), c(ff), self.dropout),
                            DecoderLayer(self.d_model, self.tgt_len, c(attn), c(attn), c(ff), self.dropout),
                                          self.N, self.d_latent, self.bypass_bottleneck)
        # Generator
        generator = Generator(self.d_model, self.vocab_size)

        # Model
        self.model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

        # Initialize model parameters
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.use_gpu = torch.cuda.is_available()
        self.model.to(self.device)
        self.char_weights = torch.tensor(self.char_weights).float().to(self.device)

        # Construct dictionaries for syntax masking
        # True: token is masked (not allowed), False: token is allowed
        bracket_begin_list = list(range(190, 308)) # [...
        bracket_end_list = (
            list(range(88, 103)) + # +...]
            list(range(104, 112)) + # -...]
            [113] + # -]
            list(range(127, 139)) + # @...]
            list(range(144, 185)) + # H...]
            [309, 318] # ], <unk>
        )
        bracket_end_list_2 = (
            list(range(88, 103)) + # +...]
            list(range(104, 112)) + # -...]
            [113] + # -]
            list(range(127, 139)) + # @...]
            list(range(144, 185)) # H...]
        )
        dash_list = [103] # -
        bond_list = [1, 126] # #, =
        number_simple_list = list(range(116, 125)) # 1, 2, ...,
        number_with_percent_list = list(range(2, 85)) # %10, %11, ...,
        end_list = [319] # <end>
        parenthesis_end_list = [86] # )
        cannot_end_start_list = [114, 112, 115, 125, 308, 317, 316] # ., ->, /, <-, \, _, ~
        
        self.dict_rules_allow = {i-1: [j-1 for j in bracket_end_list]+[k-1 for k in number_simple_list]+[d-1 for d in dash_list] for i in bracket_begin_list} # if token is in bracket_begin_list, it must be followed by a token in bracket_end_list or number_simple_list
        self.dict_rules_disallow = {115-1: [j-1 for j in bracket_end_list], # / not followed by any token in bracket_end_list
                                    308-1: [j-1 for j in bracket_end_list], # \ not followed by any token in bracket_end_list
                                    114-1: [j-1 for j in bracket_end_list]+[126-1, 103-1]+[k-1 for k in number_simple_list]+[l-1 for l in number_with_percent_list], # . not followed by any token in bracket_end_list, =, -, or any number token
                                    85-1: [j-1 for j in bracket_end_list]+[k-1 for k in number_simple_list]+[l-1 for l in number_with_percent_list]+[e-1 for e in end_list], # ( not followed by any token in bracket_end_list, or any number token
                                    86-1: [j-1 for j in bracket_end_list], # ) not followed by any token in bracket_end_list
                                    126-1: [j-1 for j in bracket_end_list]+[d-1 for d in dash_list]+[b-1 for b in bond_list]+[e-1 for e in end_list], # = not followed by any token in bracket_end_list, or any bond token, end
                                    1-1: [j-1 for j in bracket_end_list]+[d-1 for d in dash_list]+[b-1 for b in bond_list]+[e-1 for e in end_list], # # not followed by any token in bracket_end_list, or any bond token
                                    103-1: [j-1 for j in bracket_end_list_2]+[d-1 for d in dash_list]+[b-1 for b in bond_list]+[e-1 for e in end_list], # - not followed by any token in bracket_end_list_2, or any bond token, end
                                    } 
        self.dict_rules_disallow_2 = {i-1: [j-1 for j in bracket_end_list_2] for i in number_simple_list}
        self.dict_rules_disallow_3 = {i-1: [j-1 for j in bracket_end_list_2] for i in number_with_percent_list}
        self.dict_rules_disallow_4 = {i-1: [j-1 for j in bracket_end_list_2] for i in bracket_end_list_2} # if token is in bracket_end_list_2, it cannot be followed by any token in bracket_end_list_2
        self.dict_rules_disallow = {**self.dict_rules_disallow, **self.dict_rules_disallow_2, **self.dict_rules_disallow_3, **self.dict_rules_disallow_4}
        self.dict_mask_disallow = torch.zeros((self.vocab_size-1, self.vocab_size-1), dtype=torch.bool)
        for token_id, allowed_ids in self.dict_rules_allow.items():
            disallow = torch.ones(self.vocab_size-1, dtype=torch.bool)
            disallow[allowed_ids] = False
            self.dict_mask_disallow[token_id] = disallow
        for token_id, disallow_ids in self.dict_rules_disallow.items():
            disallow = torch.zeros(self.vocab_size-1, dtype=torch.bool)
            disallow[disallow_ids] = True
            self.dict_mask_disallow[token_id] = self.dict_mask_disallow[token_id] | disallow
        self.dict_mask_disallow = self.dict_mask_disallow.to(self.device)
        
        # Create mask for <start> token to disallow any token that cannot start a valid SMILES string (e.g. cannot start with a closing bracket, a bond token, or a number token)
        disallow = torch.zeros(self.vocab_size-1, dtype=torch.bool)
        for token_id in bracket_end_list + number_simple_list + dash_list + bond_list + number_with_percent_list + end_list + parenthesis_end_list + cannot_end_start_list:
            disallow[token_id-1] = True
        self.dict_mask_disallow_start = disallow.to(self.device)

        # print("MODEL ARCHITECTURE\n", self.model)

        if mode == "training":
            ### Initiate optimizer
            print("Initializing optimizer...")
            self.optimizer = NoamOpt(self.d_model, 
                                    self.lr_scale, 
                                    self.warmup_steps,
                                    torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))
            print("MODEL OPTIMIZER\n", self.optimizer)
            ### Initialize Annealer
            print("Initializing KL Annealer...")
            self.kl_annealer = KLAnnealer(self.beta_init, 
                                        self.beta, 
                                        self.kl_n_epoch,
                                        self.anneal_start)
            print("KL ANNEALER\n", self.kl_annealer)

    def load(self, checkpoint_path, mode="inference", finetune=False):
        """
        Loads a saved model state

        Arguments:
            mode (str, required): "training" or "inference"
            checkpoint_path (str, required): Path to saved .ckpt file
            finetune (bool): If true, loads model for finetuning
        """
        loaded_checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.loaded_from = checkpoint_path
        if finetune:
            self.args.__dict__['finetune_from'] = checkpoint_path
        
        if not finetune:
            # load all training information for resuming training
            for k in self.current_state.keys():
                try:
                    self.current_state[k] = loaded_checkpoint[k]
                except KeyError:
                    self.current_state[k] = None
            self.name = loaded_checkpoint['name']
        else:
            # only load model state dict for finetuning
            self.current_state['model_state_dict'] = loaded_checkpoint['model_state_dict']

        if not finetune:
            # load training information for resuming training
            self.n_epochs = self.current_state['epoch']
            self.best_loss = self.current_state['best_loss']
            self.best_epoch = self.current_state['best_epoch'] if 'best_epoch' in self.current_state.keys() else 0
            
        if not finetune:
            # load all architecture and training params for resuming training
            for k, v in self.current_state['args'].__dict__.items():
                if k in self.arch_params or k not in self.args.__dict__.keys():
                    self.args.__dict__[k] = v
                else:
                    pass
        else:
            # only load architecture params for finetuning
            for k, v in loaded_checkpoint['args'].__dict__.items():
                # only architecture and args from pretrained model
                if k in self.arch_params or k in self.args_form_pretrained: 
                    self.args.__dict__[k] = v
                # use input args
                elif k not in self.args_inputs: 
                    self.args.__dict__[k] = v
                else:
                    pass
        self.current_state['args'] = self.args
        
        self.build_model(mode=mode)
        self.model.load_state_dict(self.current_state['model_state_dict'])
        # print("Loaded checkpoint from epoch {} with val loss {}".format(self.n_epochs, self.best_loss))
        
        if mode == "training" and not finetune:
            self.optimizer.load_state_dict(self.current_state['optimizer_state_dict'])
            self.kl_annealer.load_state_dict(self.current_state['kl_annealer_state_dict'])
            print("optimizer state dict:", self.optimizer._rate, self.optimizer._step)
            print("kl_annealer state dict:", self.kl_annealer.state_dict())
            if not finetune:
                self.n_epochs += 1
                print("Continuing training from epoch {}...".format(self.n_epochs))

        