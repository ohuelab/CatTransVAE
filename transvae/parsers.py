import argparse
import numpy as np
import random
import torch
from transvae.training_mol import TransVAE as TransVAE_Mol

def device_init(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("warn")
    # torch.autograd.set_detect_anomaly(True) # make training slower
    return device

def model_init(args, mode="training", load_fn=None, finetune=False, property_external=False):
    ### Model Name
    if args.save_name is None:
        save_name = '{}-{}_{}_{}'.format(args.seed, args.model_type, args.d_latent, args.data_source)
    else:
        save_name = args.save_name

    ### Load Model
    if args.model_type == 'transvae':
        vae = TransVAE_Mol(args=args, name=save_name, mode=mode, load_fn=load_fn, finetune=finetune)
    return vae

def vocab_parser():
    parser = argparse.ArgumentParser()
    ### Vocab Parameters
    parser.add_argument('--data_type', choices=['mol'], required=True, type=str)
    parser.add_argument('--data_source', required=True, type=str)
    parser.add_argument('--max_len', default=300, type=int)
    parser.add_argument('--freq_penalty', default=0.5, type=float)
    parser.add_argument('--pad_penalty', default=0.1, type=float)
    parser.add_argument('--vocab_name', default='custom_char_dict', type=str)
    parser.add_argument('--weights_name', default='custom_char_weights', type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    return parser

def train_parser_mol():
    parser = argparse.ArgumentParser()
    ### Architecture Parameters
    parser.add_argument('--model_type', choices=['transvae'], required=True, type=str)
    parser.add_argument('--d_model', default=256, type=int)
    parser.add_argument('--d_ff', default=512, type=int)
    parser.add_argument('--d_latent', default=512, type=int)
    parser.add_argument('--eps_scale', default=1, type=float)
    parser.add_argument('--N', default=3, type=int)
    parser.add_argument('--h', default=8, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    ### Hyperparameters
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--batch_chunks', default=1, type=int)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--beta_init', default=1e-8, type=float)
    parser.add_argument('--kl_n_epoch', default=100, type=int)
    parser.add_argument('--anneal_start', default=0, type=int)
    parser.add_argument('--adam_lr', default=1e-3, type=float)
    parser.add_argument('--lr_scale', default=1, type=float)
    parser.add_argument('--warmup_steps', default=100000, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    ### Data Parameters
    parser.add_argument('--max_len', default=300, type=int)
    parser.add_argument('--data_source', required=True, type=str)
    parser.add_argument('--vocab_path', default=None, type=str)
    parser.add_argument('--char_weights_path', default=None, type=str)
    parser.add_argument('--expansion', default="none", type=str)
    parser.add_argument('--augmentation', default=0, type=int)
    ### Load Parameters
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--finetune', default='false', type=str)
    ### Save Parameters
    parser.add_argument('--save_start', default=0, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--save_name', default=None, type=str)

    return parser

def sample_parser_mol():
    parser = argparse.ArgumentParser()
    ### Load Files
    parser.add_argument('--model_type', choices=['transvae'], required=True, type=str)
    parser.add_argument('--checkpoint', required=True, type=str)
    # parser.add_argument('--vocab_path', default=None, type=str)
    parser.add_argument('--data_source', required=True, type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    ### Sampling Parameters
    parser.add_argument('--sample_mode', choices=['rand', 'high_entropy', 'k_high_entropy','rand_training', 'rand_target'], required=True, type=str)
    parser.add_argument('--decode_method', choices=['greedy', 'beam'], required=True, type=str, default='greedy')
    parser.add_argument('--k_entropy', default=15, type=int)
    parser.add_argument('--prompt', default='', type=str, required=False)
    parser.add_argument('--entropy_cutoff', default=5, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--top_k', default=-1, type=int)
    parser.add_argument('--do_sample', default='false', type=str)
    parser.add_argument('--n_samples', default=10, type=int)
    parser.add_argument('--n_samples_per_batch', default=100, type=int)
    parser.add_argument('--dummy_attaches_enabled', default='true', type=str)
    ### Save Parameters
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save_name', default=None, type=str)

    return parser
    

def prediction_parser_mol():
    parser = argparse.ArgumentParser()
    ### Load Files
    parser.add_argument('--model_type', choices=['transvae'], required=True, type=str)
    parser.add_argument('--checkpoint', required=True, type=str)
    # parser.add_argument('--vocab_path', default=None, type=str)
    parser.add_argument('--data_source', required=True, type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    ### Sampling Parameters
    parser.add_argument('--sample_mode', choices=['rand', 'high_entropy', 'k_high_entropy','rand_training', 'rand_target'], required=False, type=str)
    parser.add_argument('--decode_method', choices=['greedy', 'beam'], required=False, type=str, default='greedy')
    parser.add_argument('--k_entropy', default=15, type=int)
    parser.add_argument('--prompt', default='', type=str, required=False)
    parser.add_argument('--entropy_cutoff', default=5, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--top_k', default=-1, type=int)
    parser.add_argument('--do_sample', default='false', type=str)
    parser.add_argument('--n_samples', default=100, type=int)
    parser.add_argument('--n_samples_per_batch', default=10, type=int)
    parser.add_argument('--dummy_attaches_enabled', default='true', type=str)
    ### Save Parameters
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save_name', default=None, type=str)
    ### Prediction Parameters
    parser.add_argument('--prediction_model_type', required=False, type=str)
    parser.add_argument('--prediction_dataset', required=True, type=str)
    parser.add_argument('--prediction_embeddings', required=True, type=str)

    return parser


def optimization_parser_mol():
    parser = argparse.ArgumentParser()
    ### Load Files
    parser.add_argument('--model_type', choices=['transvae'], required=True, type=str)
    parser.add_argument('--checkpoint_gen', required=True, type=str)
    # parser.add_argument('--vocab_path', default=None, type=str)
    parser.add_argument('--data_source', required=True, type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    ### Sampling Parameters
    parser.add_argument('--sample_mode', choices=['rand', 'high_entropy', 'k_high_entropy','rand_training', 'rand_target'], required=True, type=str)
    parser.add_argument('--decode_method', choices=['greedy', 'beam'], required=True, type=str, default='greedy')
    parser.add_argument('--k_entropy', default=15, type=int)
    parser.add_argument('--prompt', default='', type=str, required=False)
    parser.add_argument('--entropy_cutoff', default=5, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--top_k', default=-1, type=int)
    parser.add_argument('--do_sample', default='false', type=str)
    parser.add_argument('--n_samples', default=10, type=int)
    parser.add_argument('--n_samples_per_batch', default=100, type=int)
    parser.add_argument('--dummy_attaches_enabled', default='true', type=str)
    ### Save Parameters
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save_name', default=None, type=str)
    ### Prediction Parameters
    parser.add_argument('--prediction_model_type', required=False, type=str)
    parser.add_argument('--prediction_dataset', required=True, type=str)
    parser.add_argument('--prediction_embeddings', required=True, type=str)
    parser.add_argument('--checkpoint_pred', required=True, type=str)

    return parser
