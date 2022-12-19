# Plug & Play Directed Evolution

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

## Sampler implementation

- product of experts: `./src/energy.py`
- PPDE MCMC sampler: `./src/protein_samplers/ppde.py`
- models: `./src/nets.py`