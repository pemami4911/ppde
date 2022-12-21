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

install ESM2 with one-hot encoded proteins:

```bash
git clone https://github.com/pemami4911/esm_one_hot
cd esm_one_hot/
pip install -e .
```

## Sampler implementation key files

- product of experts: `./src/energy.py`
- PPDE MCMC sampler: `./src/protein_samplers/ppde.py`
- models: `./src/nets.py`
- launch file: `protein_optimization.py`

## Cite the work

@article{emami2022plug,
 author = {Emami, Patrick and Perreault, Aidan and Law, Jeffrey and Biagioni, David and St John, Peter C.},
 journal = {ArXiv preprint},
 title = {Plug & Play Directed Evolution of Proteins with Gradient-based Discrete MCMC},
 url = {https://arxiv.org/abs/2212.09925,
 volume = {abs/2212.09925},
 year = {2022}
}
