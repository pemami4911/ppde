# Plug & Play Directed Evolution

A fast MCMC-based sampler for mixing and matching unsupervised and supervised sequence models for machine-learning-based directed evolution of proteins.

For now, check out this [Colab demo for running the protein samplers](https://colab.research.google.com/drive/1s3heukQga1ShfxrAMRxNtZFfSwu_D_m7?usp=sharing).

[Link to arxiv](https://arxiv.org/abs/2212.09925)

THIS REPOSITORY IS UNDER CONSTRUCTION. CODE IS PROVIDED AS IS UNTIL FURTHER NOTICE.

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

## Run MNIST samplers

Simulated annealing sampler
```
python3 scripts/mnist_sum.py --seed 1 --sampler simulated_annealing --prior ebm --energy_function joint --simulated_annealing_temp 10 --muts_per_seq_param 5 --energy_lamda 30 --n_iters 20000 --log_every 50 --wild_type 1
```

MALA-approx sampler
```
python3 scripts/mnist_sum.py --seed 1 --sampler MALA-approx --prior ebm --energy_function joint --diffusion_step_size 0.1 --diffusion_relaxation_tau 0.9 --energy_lamda 5 --n_iters 20000 --log_every 50 --wild_type 1
```

CMA-ES sampler
```
python3 scripts/mnist_sum.py --seed 1 --sampler CMAES --prior ebm --energy_function joint --energy_lamda 20 --cmaes_initial_variance 0.1 --n_iters 20000 --log_every 50 --wild_type 1
```

PPDE sampler
```
python3 scripts/mnist_sum.py --seed 1 --sampler PPDE --prior ebm --energy_function joint --ppde_pas_length 10 --energy_lamda 10 --n_iters 20000 --log_every 50 --wild_type 1
```            

## Protein sampler implementation files

Protein sampling experiment scripts are under construction. 
For now, check out the relevant source files here: 

- product of experts: `./src/energy.py`
- PPDE MCMC sampler: `./src/protein_samplers/ppde.py`
- models: `./src/nets.py`
- launch file: `protein_optimization.py`

## Cite the work

```
@article{emami2022plug,
 author = {Emami, Patrick and Perreault, Aidan and Law, Jeffrey and Biagioni, David and St John, Peter C.},
 journal = {ArXiv preprint},
 title = {Plug & Play Directed Evolution of Proteins with Gradient-based Discrete MCMC},
 url = {https://arxiv.org/abs/2212.09925,
 volume = {abs/2212.09925},
 year = {2022}
}
```
