import os
from os import path
import torch
import numpy as np
from ppde.nets import EnsembleMNIST, MNISTRegressionNet, OnehotCNN, PottsTransformer
from ppde.nets import PottsModel, EnsembleProtein
from ppde.nets import DAE
from ppde.nets import Transformer
from ppde.third_party.grathwohl import mlp
from ppde.third_party.hsu import io_utils, data_utils
    
    
class MNISTProductOfExperts(torch.nn.Module):
    def __init__(self, ebm_init_mean, args):
        super().__init__()
        self.lamda = args.energy_lamda
        self.supervised_expert = EnsembleMNIST([args.mnist_weights / f'ensemble_{i}_ckpt_25000.pt' for i in range(3)],
                                      lambda x=16: MNISTRegressionNet(x), args.device )
        self.supervised_expert.eval()
        self.unsupervised_expert_type = args.unsupervised_expert

        if args.unsupervised_expert == 'ebm':
            eps = 1e-2
            ebm_init_mean = ebm_init_mean * (1. - 2 * eps) + eps
            net = mlp.ResNetEBM()
            self.unsupervised_expert = mlp.EBM(net, ebm_init_mean)
            self.unsupervised_expert.load_state_dict(torch.load(
                args.mnist_weights / 'mnist_ebm.pt',
                map_location=torch.device(args.device))['model'])
            self.unsupervised_expert.eval()
        elif args.unsupervised_expert == 'dae':
            self.unsupervised_expert = DAE(latent_dim=16, n_channels=64, max_p=15)
            self.unsupervised_expert.to(args.device)
            self.unsupervised_expert.load_state_dict(torch.load(
                 args.mnist_weights / 'mnist_binary_dae.pt', 
                 map_location=torch.device(args.device))['model'])
            self.unsupervised_expert.eval()


    def get_energy(self, x2, **kwargs):
        x1 = kwargs['x1']
        fit = self.supervised(x1, x2)
        if self.unsupervised_expert_type == 'ebm':
            return self.unsupervised_expert(x2) + self.lamda * fit, fit
        elif self.unsupervised_expert_type == 'dae':
            logp = self.unsupervised_expert.log_prob(x2)
            return logp + self.lamda * fit, fit
        
    def get_supervised_expert(self, x2, **kwargs):
        x1 = kwargs['x1']
        return self.supervised_expert(x1, x2)


class MNISTSupervised(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.supervised_expert = EnsembleMNIST([args.mnist_weights / f'ensemble_{i}_ckpt_25000.pt' for i in range(3)],
                                      lambda x=16: MNISTRegressionNet(x), args.device )
        self.supervised_expert.eval()

    def get_energy(self, x2, **kwargs):
        x1 = kwargs['x1']
        fit = self.supervised_expert(x1, x2)
        return fit, fit
        
    def get_supervised_expert(self, x2, **kwargs):
        x1 = kwargs['x1']
        return self.supervised_expert(x1, x2)


class ProteinProductOfExperts(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lamda = args.energy_lamda
        self.unsupervised_expert_type = args.unsupervised_expert
        dataset = path.join(args.protein_weights, args.protein)
        self.minibatch_size = 8 if args.unsupervised_expert == 'transformer-L' else min(args.n_chains,64)
        #self.minibatch_size = args.n_chains

        if args.unsupervised_expert == 'potts':
            self.unsupervised_expert = PottsModel(dataset)
            self.unsupervised_expert.eval()
        elif args.unsupervised_expert == 'potts+transformer':
            self.unsupervised_expert = PottsTransformer(dataset)
            self.unsupervised_expert.eval()
        elif 'transformer' in args.unsupervised_expert:
            self.unsupervised_expert_type = 'transformer'
            self.unsupervised_expert = Transformer(args.unsupervised_expert, dataset)
            self.unsupervised_expert.eval()

        wt = io_utils.read_fasta(path.join(dataset,'wt.fasta'), return_ids=False)
        self.supervised_expert = EnsembleProtein([path.join(dataset, f'onehot_cnn_seed={i}.pt') for i in range(3)],
                                      lambda x=20, y=5, z=len(wt[0]), : OnehotCNN(x,y,z), args.device )
        self.supervised_expert.eval()
        self.wt_onehot = torch.from_numpy(data_utils.seqs_to_onehot(wt)).float().to(args.device)

    def get_energy(self, x):
        fit = self.supervised_expert(x)
        e = self.unsupervised_expert(self.unsupervised_expert.preprocess_onehot(x), delta=True) \
            + self.lamda * fit
        return e, fit

    def get_energy_and_grads(self, x):
        fit = self.supervised_expert(x)
        if self.unsupervised_expert_type == 'potts':
            e = self.unsupervised_expert(self.unsupervised_expert.preprocess_onehot(x), delta=True) \
                + self.lamda * fit
            grad_x = torch.autograd.grad([e.sum()], x)[0]

        elif self.unsupervised_expert_type == 'transformer' or \
            self.unsupervised_expert_type == 'potts+transformer':
            all_grads = []
            all_e = []
            num_minibatches = int(np.ceil(x.size(0) / self.minibatch_size))
            for i in range(num_minibatches):
                x_batch = x[i*self.minibatch_size : (i+1)*self.minibatch_size]
                unsupervised_expert = self.unsupervised_expert(
                    self.unsupervised_expert.preprocess_onehot(x_batch), delta=True)
                if self.minibatch_size > 1:
                    fit_ = fit[i*self.minibatch_size : (i+1)*self.minibatch_size]
                else:
                    fit_ = fit
                e = unsupervised_expert + self.lamda * fit_
                all_e += [e.detach()]
                all_grads += [torch.autograd.grad([e.sum()], x_batch)[0]]

                self.unsupervised_expert.zero_grad()

            e = torch.cat(all_e, 0)
            grad_x = torch.cat(all_grads, 0)

        return e, fit, grad_x

    def get_supervised_expert(self, x):
        fit = self.supervised_expert(x)
        return fit

    def get_unsupervised_expert(self, x):
        return self.unsupervised_expert(
            self.unsupervised_expert.preprocess_onehot(x), delta=True)


class ProteinSupervised(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        dataset = path.join(args.protein_weights, args.protein)
        wt = io_utils.read_fasta(path.join(dataset,'wt.fasta'), return_ids=False)
        self.supervised_expert = EnsembleProtein([path.join(dataset, f'onehot_cnn_seed={i}.pt') for i in range(3)],
                                      lambda x=20, y=5, z=len(wt[0]), : OnehotCNN(x,y,z), args.device )
        self.supervised_expert.eval()
        self.wt_onehot = torch.from_numpy(data_utils.seqs_to_onehot(wt)).float().to(args.device) 

    def get_energy(self, x):
        fit = self.supervised_expert(x)
        return fit, fit

    def get_energy_and_grads(self, x):
        fit = self.supervised_expert(x)
        grad_x = torch.autograd.grad([fit.sum()], x)[0]
        return fit, fit, grad_x

    def get_supervised_expert(self, x):
        fit = self.supervised_expert(x)        
        return fit

