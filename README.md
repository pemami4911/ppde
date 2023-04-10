# Plug & Play Directed Evolution

A fast MCMC-based sampler for mixing and matching unsupervised and supervised sequence models for machine-learning-based directed evolution of proteins.

Check out this [Colab demo for running the protein samplers](https://colab.research.google.com/drive/1s3heukQga1ShfxrAMRxNtZFfSwu_D_m7?usp=sharing).

[DOI](https://doi.org/10.1088/2632-2153/accacd) [arxiv link](https://arxiv.org/abs/2212.09925)

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
python3 scripts/mnist_sum.py --seed 1 --sampler simulated_annealing --prior ebm --energy_function product_of_experts --simulated_annealing_temp 10 --muts_per_seq_param 5 --energy_lamda 30 --n_iters 20000 --log_every 50 --wild_type 1
```

MALA-approx sampler
```
python3 scripts/mnist_sum.py --seed 1 --sampler MALA-approx --prior ebm --energy_function product_of_experts --diffusion_step_size 0.1 --diffusion_relaxation_tau 0.9 --energy_lamda 5 --n_iters 20000 --log_every 50 --wild_type 1
```

CMA-ES sampler
```
python3 scripts/mnist_sum.py --seed 1 --sampler CMAES --prior ebm --energy_function product_of_experts --energy_lamda 20 --cmaes_initial_variance 0.1 --n_iters 20000 --log_every 50 --wild_type 1
```

PPDE sampler
```
python3 scripts/mnist_sum.py --seed 1 --sampler PPDE --prior ebm --energy_function product_of_experts --ppde_pas_length 10 --energy_lamda 10 --n_iters 20000 --log_every 50 --wild_type 1
``` 

By default, the script will save metrics and visualizations to `results/mnist_sum/`.

## Run Protein experiments

| protein | unsupervised expert | $\lambda$ | 
| --- | --- | --- |
| PABP_YEAST_Fields2013 | potts | 5 |
| UBE4B_MOUSE_Klevit2013-nscor_log2_ratio | potts | 0.5 |
| GFP_AEQVI_Sarkisyan2016 | potts | 15 |
| PABP_YEAST_Fields2013 | transformer | 5 |
| UBE4B_MOUSE_Klevit2013-nscor_log2_ratio | transformer | 3 |
| GFP_AEQVI_Sarkisyan2016 | transformer | 1 |


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

By default, the script will save metrics in `.npy` format to `results/proteins/$PROTEIN`.

## Protein sampler implementation files

Protein sampling experiment scripts are under construction. 
For now, check out the relevant source files here: 

- product of experts: `./src/energy.py`
- PPDE MCMC sampler: `./src/protein_samplers/ppde.py`
- models: `./src/nets.py`
- launch file: `protein_optimization.py`

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
