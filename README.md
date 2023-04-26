# Plug & Play Directed Evolution

A fast MCMC-based sampler for mixing and matching unsupervised and supervised sequence models for machine-learning-based directed evolution of proteins.

[[DOI](https://doi.org/10.1088/2632-2153/accacd)] [[arxiv link](https://arxiv.org/abs/2212.09925)]

## Install

Create the conda env with necessary dependencies:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate ppde
```

Install the package:

```bash
poetry install
```

## Run MNIST experiments

Simulated annealing sampler
```
python3 scripts/mnist_sum.py --seed 1 --sampler simulated_annealing --unsupervised_expert ebm --energy_function product_of_experts --simulated_annealing_temp 10 --muts_per_seq_param 5 --energy_lamda 30 --n_iters 20000 --log_every 50 --wild_type 1
```

MALA-approx sampler
```
python3 scripts/mnist_sum.py --seed 1 --sampler MALA-approx --unsupervised_expert ebm --energy_function product_of_experts --diffusion_step_size 0.1 --diffusion_relaxation_tau 0.9 --energy_lamda 5 --n_iters 20000 --log_every 50 --wild_type 1
```

CMA-ES sampler
```
python3 scripts/mnist_sum.py --seed 1 --sampler CMAES --unsupervised_expert ebm --energy_function product_of_experts --energy_lamda 20 --cmaes_initial_variance 0.1 --n_iters 20000 --log_every 50 --wild_type 1
```

PPDE sampler
```
python3 scripts/mnist_sum.py --seed 1 --sampler PPDE --unsupervised_expert ebm --energy_function product_of_experts --ppde_pas_length 10 --energy_lamda 10 --n_iters 20000 --log_every 50 --wild_type 1
``` 

By default, the script will save metrics and visualizations to `results/mnist_sum/`.

### Training MNIST models

See `./scripts/train_mnist.sh` for instructions on training the MNIST models.
Script for training the Denoising Autoencoder (DAE) model: `./scripts/train_binary_mnist_dae.py`.
Script for training the supervised experts: `./scripts/train_binary_mnist_regression.py`.

## Run Protein experiments

**UPDATE:** We found that the PPDE protein sampler was unintentially ran with a "soft" maximum number of mutations from wild type, i.e., the sampler would reset the Markov chain to the wild type whenever a mutation proposal was rejected. We have corrected the accept/reject step code (L77 in ppde/protein_samplers/ppde.py) and implemented a proper "hard" maximum number of mutations constraint. See [this PDF](data/PPDE_Updated_Results.pdf) for updated versions of Table 1 and Table 2 with results from running PPDE with a "hard" maximum of 10 mutations (this can be set with argument `--nmut_threshold`). If aiming to replicate the PPDE protein experiments results from the paper, set the flag `--paper_results` when running to use the "soft" maximum number of mutations constraint. This flag only affects the PPDE protein sampler (not the baselines or the MNIST experiments).

| protein | unsupervised expert | $\lambda$ | 
| --- | --- | --- |
| PABP_YEAST_Fields2013 | potts | 5 |
| UBE4B_MOUSE_Klevit2013-nscor_log2_ratio | potts | 0.5 |
| GFP_AEQVI_Sarkisyan2016 | potts | 15 |
| PABP_YEAST_Fields2013 | transformer | 5 |
| UBE4B_MOUSE_Klevit2013-nscor_log2_ratio | transformer | 3 |
| GFP_AEQVI_Sarkisyan2016 | transformer | 1 |

See `./scripts/run_protein_samplers.sh` or:

Random sampler
```
python3 scripts/directed_evolution.py --seed 1 --sampler Random --unsupervised_expert potts --energy_function product_of_experts --energy_lamda 5 --n_iters 10000 --log_every 50 --protein PABP_YEAST_Fields2013 --msa_path data/proteins/PABP_YEAST.a2m
```

Simulated annealing sampler
```
python3 scripts/directed_evolution.py --seed 1 --sampler simulated_annealing --unsupervised_expert potts --energy_function product_of_experts --energy_lamda 5 --n_iters 10000 --log_every 50 --protein PABP_YEAST_Fields2013 --msa_path data/proteins/PABP_YEAST.a2m
```

MALA-approx sampler
```
python3 scripts/directed_evolution.py --seed 1 --sampler MALA-approx --unsupervised_expert potts --energy_function product_of_experts --energy_lamda 5 --n_iters 10000 --log_every 50 --protein PABP_YEAST_Fields2013 --msa_path data/proteins/PABP_YEAST.a2m
```

CMA-ES sampler
```
python3 scripts/directed_evolution.py --seed 1 --sampler CMAES --unsupervised_expert potts --energy_function product_of_experts --energy_lamda 5 --n_iters 1000 --log_every 50 --protein PABP_YEAST_Fields2013 --msa_path data/proteins/PABP_YEAST.a2m
```

PPDE sampler
```
python3 scripts/directed_evolution.py --seed 1 --sampler PPDE --unsupervised_expert potts --energy_function product_of_experts --energy_lamda 5 --n_iters 100 --log_every 50 --protein PABP_YEAST_Fields2013 --msa_path data/proteins/PABP_YEAST.a2m
``` 

By default, the script will save metrics in `.npy` format to `results/proteins/$PROTEIN`. Compute metrics with `./scripts/make_figures.py`.

## Cite the work

```
@article{emami2023plug,
	author={Emami, Patrick and Perreault, Aidan and Law, Jeffrey and Biagioni, David and St. John, Peter C},
	title={Plug & Play Directed Evolution of Proteins with Gradient-based Discrete MCMC},
	journal={Machine Learning: Science and Technology},
	url={http://iopscience.iop.org/article/10.1088/2632-2153/accacd},
	year={2023}
}
```
