import os
from os import path
import torch
import numpy as np
from src.nets import EnsembleMNIST, MNISTRegressionNet, MNISTLatentMLP, OnehotCNN, PottsTransformer
from src.nets import PottsModel, EnsembleProtein
from src.nets import DAE
from src.nets import Transformer
from src.third_party.grathwohl import mlp
from src.third_party.hsu import io_utils, data_utils
    
class MNISTJointEnergy(torch.nn.Module):
    def __init__(self, ebm_init_mean, args):
        super().__init__()
        self.lamda = args.energy_lamda

        self.fitness = EnsembleMNIST([path.join(args.data_dir,args.one_hot_ensemble_path,f'ensemble_{i}_ckpt_25000.pt') for i in range(3)],
                                      lambda x=16: MNISTRegressionNet(x), args.device )
        self.fitness.eval()
        self.prior_type = args.prior

        if args.prior == 'ebm':
            eps = 1e-2
            ebm_init_mean = ebm_init_mean * (1. - 2 * eps) + eps
            net = mlp.ResNetEBM()
            self.prior = mlp.EBM(net, ebm_init_mean)
            self.prior.load_state_dict(torch.load(path.join(args.data_dir, args.ebm_path))['model'])
            self.prior.eval()
        elif args.prior == 'dae':
            self.prior = DAE(latent_dim=16, n_channels=64, max_p=15)
            self.prior.to(args.device)
            self.prior.load_state_dict(torch.load(path.join(args.data_dir, args.dae_path))['model'])
            self.prior.eval()


    def get_energy(self, x2, **kwargs):
        x1 = kwargs['x1']
        fit = self.fitness(x1, x2)
        if self.prior_type == 'ebm':
            return self.prior(x2) + self.lamda * fit, fit
        elif self.prior_type == 'dae':
            logp = self.prior.log_prob(x2)
            return logp + self.lamda * fit, fit
        
    def get_fitness(self, x2, **kwargs):
        x1 = kwargs['x1']
        return self.fitness(x1, x2)


class MNISTEnergy(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fitness = EnsembleMNIST([path.join(args.data_dir,args.one_hot_ensemble_path,f'ensemble_{i}_ckpt_25000.pt') for i in range(3)],
                                      lambda x=16: MNISTRegressionNet(x), args.device )
        self.fitness.eval()

    def get_energy(self, x2, **kwargs):
        x1 = kwargs['x1']
        fit = self.fitness(x1, x2)
        return fit, fit
        
    def get_fitness(self, x2, **kwargs):
        x1 = kwargs['x1']
        return self.fitness(x1, x2)


class TestMNISTEnergy(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fitness = EnsembleMNIST([path.join(args.data_dir,args.one_hot_ensemble_path,f'ensemble_{i}_ckpt_25000.pt') for i in range(3)],
                                      lambda x=16: MNISTRegressionNet(x), args.device )
        self.fitness.eval()

    def get_energy(self, x2, **kwargs):
        fit = self.get_fitness(x2)
        return fit, fit
        
    def get_fitness(self, x2, **kwargs):
        return torch.sum( x2.view(x2.size(0), 28*28), -1)


class MNISTLatentSurrogate(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fitness = EnsembleMNIST([path.join(args.data_dir, args.one_hot_ensemble_path, f'latent_ensemble_{i}_ckpt_40000.pt') for i in range(3)],
                                      lambda x=16,y=128: MNISTLatentMLP(x,y), args.device )
        self.fitness.eval()

    def get_energy(self, x2, **kwargs):
        x1 = kwargs['x1']
        fit = self.fitness(x1, x2)
        return fit, fit
        
    def get_fitness(self, x2, **kwargs):
        x1 = kwargs['x1']
        return self.fitness(x1, x2)


class ProteinJointEnergy(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lamda = args.energy_lamda
        self.prior_type = args.prior
        dataset = path.join(args.data_dir, 'weights', args.dataset_name)
        self.minibatch_size = 1 if args.prior == 'transformer-L' else min(args.n_chains,16)

        if args.prior == 'potts':
            self.prior = PottsModel(dataset)
            self.prior.eval()
        elif 'transformer-' in args.prior:
            self.prior_type = 'transformer'
            self.prior = Transformer(args.prior, dataset)
            self.prior.eval()
        elif args.prior == 'potts+transformer':
            self.prior = PottsTransformer(dataset)
            self.prior.eval()

        wt = io_utils.read_fasta(os.path.join(args.data_dir, 'weights', args.dataset_name,
            'wt.fasta'), return_ids=False)
        self.fitness = EnsembleProtein([path.join(args.data_dir, 'weights', args.dataset_name, f'onehot_cnn_seed={i}.pt') for i in range(3)],
                                      lambda x=20, y=5, z=len(wt[0]), : OnehotCNN(x,y,z), args.device )
        self.fitness.eval()
        self.wt_onehot = torch.from_numpy(data_utils.seqs_to_onehot(wt)).float().to(args.device)

    def get_energy(self, x):
        fit = self.fitness(x)
        e = self.prior(self.prior.preprocess_onehot(x), delta=True) + self.lamda * fit
        return e, fit

    def get_energy_and_grads(self, x):
        fit = self.fitness(x)
        if self.prior_type == 'potts':
            e = self.prior(self.prior.preprocess_onehot(x), delta=True) + self.lamda * fit
            grad_x = torch.autograd.grad([e.sum()], x)[0]

        elif self.prior_type == 'transformer' or self.prior_type == 'potts+transformer':
            all_grads = []
            all_e = []
            num_minibatches = int(np.ceil(x.size(0) / self.minibatch_size))
            for i in range(num_minibatches):
                x_batch = x[i*self.minibatch_size : (i+1)*self.minibatch_size]
                prior = self.prior(self.prior.preprocess_onehot(x_batch), delta=True)
                if self.minibatch_size > 1:
                    fit_ = fit[i*self.minibatch_size : (i+1)*self.minibatch_size]
                else:
                    fit_ = fit
                e = prior + self.lamda * fit_
                all_e += [e.detach()]
                all_grads += [torch.autograd.grad([e.sum()], x_batch)[0]]

                self.prior.zero_grad()

            e = torch.cat(all_e, 0)
            grad_x = torch.cat(all_grads, 0)

        return e, fit, grad_x

    def get_fitness(self, x):
        fit = self.fitness(x)
        return fit

    def get_prior(self, x):
        return self.prior(self.prior.preprocess_onehot(x), delta=True)


class ProteinEnergy(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        wt = io_utils.read_fasta(os.path.join(args.data_dir, 'weights', args.dataset_name,
            'wt.fasta'), return_ids=False)
        self.fitness = EnsembleProtein([path.join(args.data_dir, 'weights', args.dataset_name, f'onehot_cnn_seed={i}.pt') for i in range(3)],
                                      lambda x=20, y=5, z=len(wt[0]), : OnehotCNN(x,y,z), args.device )
        self.fitness.eval()
        self.wt_onehot = torch.from_numpy(data_utils.seqs_to_onehot(wt)).float().to(args.device) 

    def get_energy(self, x):
        fit = self.fitness(x)
        return fit, fit

    def get_energy_and_grads(self, x):
        fit = self.fitness(x)
        grad_x = torch.autograd.grad([fit.sum()], x)[0]
        return fit, fit, grad_x

    def get_fitness(self, x):
        fit = self.fitness(x)        
        return fit

