# CatTransVAE

Catalyst-specialized chemical language model based on a Transformer variational autoencoder (VAE) designed to improve catalyst recognition and generative performance across diverse catalyst classes through template-guided molecular design with high task-validity and diversity.

<img width="2930" height="967" alt="GraphicalAbstact-s" src="https://raw.githubusercontent.com/ohuelab/CatTransVAE/refs/heads/main/image/abstract.png" />

## Google Colab 🪄

- Quick start usage with google colab 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1M1xLBqNfNvcaIvQs-6cyF92wmSFXXc-R/view?usp=sharing)
- Include sampling, template-guied generation, and two-level embedding extraction

## Model 🦾

- TransVAE on pubchem10: [019_pubchem10M_model_20_01_10](https://science-tokyo.box.com/s/n8xau7b2b4y46kb85b4haigewbkkuzbb)
- CatTransVAE on CatalystSet_TMC_D: [039_CatalystSet_TMC_D_L_10M01901_40_01_20](https://science-tokyo.box.com/s/lasru81kuhx58prquq9qawbavvtmarec)

## Datasets 📑

- Pubchem: https://huggingface.co/datasets/hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC 
- Pubchem10: Sample 10M molecules from PubChem dataset
- CatalystSet: Original sources mentioned in paper


## Installation 🛠️

Install dependencies. This code was tested in Python 3.8 with PyTorch and rdkit.

```bash
conda create -f cattransvae.yaml
conda activate cattransvae
```

## Usage 📔

- [Build vocabulary](#build-vocabulary)
- [Pre-train foundation model](#pre-train-foundation-model)
- [Fine-tune foundation model to catalyst dataset](#fine-tune-foundation-model-to-catalyst-dataset)
- [Test reconstruction](#test-reconstruction)
- [Embedding space evaluation](#embedding-space-evaluation)
- [Sampling and generation](#sampling-and-generation)
- [Evaluate a set of sample cases](#evaluate-a-set-of-sample-cases)
- [Prediction 5-fold](#prediction-5-fold)
- [Optimization and guided generation](#optimization-and-guided-generation)

### Build vocabulary:

- This is optional process. You can access dictionary in pubchem10 folder
- Please download pubchem dataset and rename to `pubchem10M`

```bash
python 01_build_vocab.py \
--data_type mol \
--data_source pubchem10M \
--vocab_name pubchem10M/pubchem10M_dict \
--weights_name pubchem10M/pubchem10M_weight
```

### Pre-train foundation model:

- `<DATA_SOURCE>` : Dataset for pre-training e.g., pubchem10M, CatalystSet_S, CatalystSet_TMC_NoD, CatalystSet_TMC_D
- `<SAVE_NAME>` : Name of model

```bash
python 02_train_mol.py \
--model_type transvae \
--data_source <DATA_SOURCE> \
--vocab_path data/pubchem10M/pubchem10M_dict.pkl \
--char_weights_path data/pubchem10M/pubchem10M_weight.npy \
--save_name <SAVE_NAME> \
--d_model 256 \
--d_latent 512 \
--batch_size 256 \
--batch_chunks 1 \
--epochs 20 \
--beta 0.1 \
--kl_n_epoch 10 \
--warmup_steps 100000
```

#### Continue training from checkpoint:

- `<DATA_SOURCE>` : Dataset for pre-training e.g., pubchem10M, CatalystSet_S, CatalystSet_TMC_NoD, CatalystSet_TMC_D
- `<SAVE_NAME>` : Name of model
- `<CHECKPOINT>` : Previous pretrained model path `data/<DATA_SOURCE>/checkpoints/<EPOCH_TO_CONTINUE>_<DATA_SOURCE>_<SAVE_NAME>.ckpt` (e.g. data/pubchem10M/checkpoints/012_pubchem10M_model_20_05_10.ckpt)

```bash
python 02_train_mol.py \
--model_type transvae \
--data_source <DATA_SOURCE> \
--vocab_path data/pubchem10M/pubchem10M_dict.pkl \
--char_weights_path data/pubchem10M/pubchem10M_weight.npy \
--save_name <SAVE_NAME> \
--d_model 256 \
--d_latent 512 \
--batch_size 256 \
--batch_chunks 1 \
--epochs 20 \
--beta 0.1 \
--kl_n_epoch 10 \
--warmup_steps 100000
--checkpoint <CHECKPOINT>
```

### Fine-tune foundation model to catalyst dataset:

- `<DATA_SOURCE>` : Dataset for finetuning e.g., CatalystSet_S, CatalystSet_TMC_NoD, CatalystSet_TMC_D
- `<SAVE_NAME>` : Name of model
- `<CHECKPOINT>` : Previous pretrained model path `data/<DATA_SOURCE_PRETRAINED>/checkpoints/<EPOCH_BEST_PRETRAINED>_<DATA_SOURCE_PRETRAINED>_<SAVE_NAME_PRETRAINED>.ckpt` (e.g. data/pubchem10M/checkpoints/019_pubchem10M_model_20_005_10.ckpt)

```bash
python 03_train_cat.py \
--model_type transvae \
--d_model 256 \
--d_latent 512 \
--data_source <DATA_SOURCE> \
--epochs 40 \
--save_name <SAVE_NAME> \
--beta 0.1 \
--kl_n_epoch 20 \
--warmup_steps 100000 \
--checkpoint <CHECKPOINT> \
--expansion pubchem10M \
--augmentation 0 \
--finetune true
```

#### Continue fine-tuning from checkpoint:

- `<DATA_SOURCE>` : Dataset for finetuning e.g., CatalystSet_S, CatalystSet_TMC_NoD, CatalystSet_TMC_D
- `<SAVE_NAME>` : Name of model
- `<CHECKPOINT>` : Previous fine-tuned model path `data/<DATA_SOURCE>/checkpoints/<EPOCH_TO_CONTINUE>_<DATA_SOURCE>_<SAVE_NAME>.ckpt` (e.g. data/CatalystSet_TMC_D/checkpoints/028_CatalystSet_TMC_D_L_10M01901_50_01_25.ckpt)


```bash
python 03_train_cat.py \
--model_type transvae \
--d_model 256 \
--d_latent 512 \
--data_source <DATA_SOURCE> \
--epochs 40 \
--save_name <SAVE_NAME> \
--beta 0.1 \
--kl_n_epoch 20 \
--warmup_steps 100000 \
--checkpoint <CHECKPOINT> \
--expansion pubchem10M \
--augmentation 0 \
--finetune false
```

### Test reconstruction:

- `<DATA_SOURCE>` : Data source to test (test set), e.g. pubchem10M, CatalystSet_S, CatalystSet_TMC_NoD, CatalystSet_TMC_D
- `<CHECKPOINT> `: Trained model path to test `data/<DATA_SOURCE>/checkpoints/<SELECTED_EPOCH>_<DATA_SOURCE>_<SAVE_NAME>.ckpt` (e.g. data/CatalystSet_TMC_D/checkpoints/039_CatalystSet_TMC_D_L_10M01901_40_01_20.ckpt)
- `<EXPERIMENT>` : Name of experiment

```bash
python 06_test_recon.py \
--model_type transvae \
--data_source <DATA_SOURCE> \
--checkpoint <CHECKPOINT> \
--sample_mode rand \
--decode_method greedy \
--save_name <EXPERIMENT> \
```

### Embedding space evaluation:

- `<DATA_SOURCE>` : Data source to test (test set), e.g. pubchem10M, CatalystSet_S, CatalystSet_TMC_NoD, CatalystSet_TMC_D
- `<CHECKPOINT>` : Trained model path to test `data/<DATA_SOURCE>/checkpoints/<EPOCH>_<DATA_SOURCE>_<SAVE_NAME>.ckpt` (e.g. data/CatalystSet_TMC_D/checkpoints/039_CatalystSet_TMC_D_L_10M01901_40_01_20.ckpt)
- `<EXPERIMENT>` : Name of experiment

```bash
python 08_test_embeddingspace.py \
--model_type transvae \
--data_source <DATA_SOURCE> \
--checkpoint <CHECKPOINT> \
--sample_mode rand \
--decode_method greedy \
--save_name <EXPERIMENT> \
```

### Sampling and generation:

- `<DATA_SOURCE>` : Data source to test (test set), e.g. pubchem10M, CatalystSet_S, CatalystSet_TMC_NoD, CatalystSet_TMC_D
- `<CHECKPOINT>` : Trained model path to test `data/<DATA_SOURCE>/<EPOCH>_<DATA_SOURCE>_<SAVE_NAME>.ckpt` (e.g. data/CatalystSet_TMC_D/checkpoints/039_CatalystSet_TMC_D_L_10M01901_40_01_20.ckpt)
- `<PROMPT>` : `'none'` or defined prompt e.g. `'CCCC[*:1]CC([*:2])'`
- `<EXPERIMENT>` : Name of experiment
- `decode_method` : `'greedy'`, `'beam'`
- `sample_mode` :  `'rand'`, `'k_high_entropy'`, `'rand_training'`, `'rand_target'`
- `k_entropy` :  `-1` (for random), `1`, `10`, `50`, ... `512` (for k_high_entropy)
- `temperature` : `1.0` (default), or floating points
- `top_k` : used when `do_sample` is True
- `do_sample` : `'true'` (select from probability), `'false'` (select highest probability)
- `dummy_attaches_enabled` : `'true'` (enable dummy augmentation), `'false'` (use default (C) dummy)

```bash
python 07_test_sample.py \
--model_type transvae \
--data_source <DATA_SOURCE> \
--checkpoint <CHECKPOINT> \
--sample_mode rand \
--decode_method greedy \
--k_entropy -1 \
--temperature 1.0 \
--top_k 10 \
--do_sample 'true' \
--dummy_attaches_enabled 'false' \
--n_samples 10 \
--n_samples_per_batch 100 \
--prompt <PROMPT> \
--save_name <EXPERIMENT> \
```

```bash
python 07_test_sample.py \
--model_type transvae \
--data_source <DATA_SOURCE> \
--checkpoint <CHECKPOINT> \
--sample_mode k_high_entropy \
--decode_method beam \
--k_entropy 50 \
--temperature 1.0 \
--top_k 10 \
--do_sample 'true' \
--dummy_attaches_enabled 'true' \
--n_samples 10 \
--n_samples_per_batch 100 \
--prompt '[*:1]c1ccc([*:2]c2ccc([*:1])cc2)cc1' \
--save_name <EXPERIMENT> \
```

### Evaluate a set of sample cases:

This is the code for testing 6 example case studies reported in paper.
- `<DATA_SOURCE>` : Data source to test (test set), e.g. pubchem10M, CatalystSet_S, CatalystSet_TMC_NoD, CatalystSet_TMC_D
- `<CHECKPOINT>` : Trained model path to test `data/<DATA_SOURCE>/checkpoints/<EPOCH>_<DATA_SOURCE>_<SAVE_NAME>.ckpt` (e.g. data/CatalystSet_TMC_D/checkpoints/039_CatalystSet_TMC_D_L_10M01901_40_01_20.ckpt)
- `<PROMPT>` : (prompts are defined inside python file)
- `<EXPERIMENT>` : Name of experiment

```bash
python 07_test_sample_cases.py \
--model_type transvae \
--data_source <DATA_SOURCE> \
--checkpoint <CHECKPOINT> \
--sample_mode rand \
--decode_method greedy \
--k_entropy 100 \
--temperature 1.0 \
--top_k -1 \
--do_sample 'true' \
--n_samples 10 \
--n_samples_per_batch 100 \
--prompt 'none' \
--save_name <EXPERIMENT>
```

### Prediction 5-fold:

- `<DATA_SOURCE>` : Data source to test (test set), e.g. pubchem10M, CatalystSet_S, CatalystSet_TMC_NoD, CatalystSet_TMC_D
- `<CHECKPOINT>` : Trained model path to test `data/<DATA_SOURCE>/checkpoints/<EPOCH>_<DATA_SOURCE>_<SAVE_NAME>.ckpt` (e.g. data/CatalystSet_TMC_D/checkpoints/039_CatalystSet_TMC_D_L_10M01901_40_01_20.ckpt)
- `<DATASET>` : example of datasets are
suzuki_7054_split_random.csv,
suzuki_7054_split_metal.csv,
vaskas_1947_8010.csv,
vaskas_1947_2040.csv,
tepid_4703_7030.csv,
tepid_4703_scaffold_0.csv
- `<EMBEDDING>` : example of embedding are
`CatTransVAE` (for two-level embedding),
`CatTransVAE_vae` (for VAE embedding),
`CatTransVAE_emb` (for Transformer embedding),
`MorganFP` (for Morgan fingerprint)
- `<SEED>` : Seed
- `<EXPERIMENT>` : Name of experiment


```bash
python prediction/prediction_5fold.py \
--model_type transvae \
--data_source <DATA_SOURCE> \
--checkpoint <CHECKPOINT> \
--prediction_model_type xgboost \
--prediction_dataset <DATASET> \
--prediction_embeddings <EMBEDDING> \
--seed <SEED> \
--save_name <EXPERIMENT>
```

### Optimization and guided generation:

- `<DATA_SOURCE>` : Data source to test (test set), e.g. pubchem10M, CatalystSet_S, CatalystSet_TMC_NoD, CatalystSet_TMC_D
- `<CHECKPOINT_GEN>` : Trained model path to test `data/<DATA_SOURCE>/checkpoint/<EPOCH>_<DATA_SOURCE>_<SAVE_NAME>.ckpt` (e.g. data/CatalystSet_TMC_D/checkpoints/039_CatalystSet_TMC_D_L_10M01901_40_01_20.ckpt)
- `<CHECKPOINT_PRED>` : Saved trained best xgboost model folder `prediction/results/<DATASET>/<SEED>_<EMBEDDING>_<MODEL_NAME>.ckpt/<EXPERIMENT>` (e.g. prediction/results/suzuki_7054_split_random_split_1.csv/1_CatTransVAE_039_CatalystSet_TMC_D_L_10M01901_40_01_20.ckpt/20260531_222307_7815518)
- `decode_method` : `'greedy'`, `'beam'`
- `sample_mode` :  `'rand'`, `'k_high_entropy'`, `'rand_training'`, `'rand_target'`
- `k_entropy` :  `-1` (for random), `1`, `10`, `50`, ... `512` (for k_high_entropy)
- `temperature` : `1.0` (default), or floating points
- `top_k` : used when `do_sample` is True
- `do_sample` : `'true'` (select from probability), `'false'` (select highest probability)
- `dummy_attaches_enabled` : `'true'` (enable dummy augmentation), `'false'` (use default (C) dummy)
- `<PROMPT>` : `'none'` or defined prompt e.g. `'CCCC[*:1]CC([*:2])'`
- `<DATASET>` : Already trained dataset
- `<EMBEDDING>` : Already trained embedding
- `<SEED>` : Seed
- `<EXPERIMENT>` : Name of experiment

```bash
python optimization/optimization.py \
--model_type transvae \
--data_source <DATA_SOURCE> \
--checkpoint_gen <CHECKPOINT_GEN> \
--checkpoint_pred <CHECKPOINT_PRED> \
--sample_mode $sample_mode \
--decode_method $decode_method \
--k_entropy $k_entropy \
--temperature $temperature \
--top_k $top_k \
--do_sample $do_sample \
--dummy_attaches_enabled $dummy_attaches_enabled \
--n_samples 1 \
--n_samples_per_batch 1 \
--prompt <PROMPT> \
--prediction_model_type xgboost \
--prediction_dataset <DATASET> \
--prediction_embeddings <EMBEDDING> \
--seed <SEED> \
--save_name <EXPERIMENT>
```

## Citation

Thank you for your interests, please kindly cite:

```bibtex
TBA...
```