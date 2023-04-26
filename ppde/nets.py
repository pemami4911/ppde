import os
import torch
from torch import nn
import torch.nn.functional as F
import math
from ppde.third_party.grathwohl.mlp import Swish
from ppde.third_party.grathwohl.mlp import BasicBlock
from ppde.third_party.hsu import io_utils, data_utils
import pickle as pkl

from esm_one_hot import pretrained


class MNISTRegressionNet(nn.Module):
    """A simple Siamese regression network for predicting 
    the sum of two MNIST digits.
    """
    def __init__(self, nc=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, nc, 4, 2, 1),
            Swish(),
            nn.Conv2d(nc, nc, 4, 2, 1),
            Swish(),
            nn.Conv2d(nc, nc, 4, 2, 1),
            Swish(),
            nn.Conv2d(nc, nc, 3, 1, 0),
            Swish()
        )
        self.out = nn.Linear(nc, 1)

    def forward(self, input1, input2):
        input1 = input1.view(input1.size(0), 1, 28, 28)
        input2 = input2.view(input2.size(0), 1, 28, 28)
        h1 = self.net(input1)
        h2 = self.net(input2)
        return self.out((h1+h2).squeeze()).squeeze()
    
    
# class MNISTLatentMLP(nn.Module):
#     def __init__(self, z=16, nh=32):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(z, nh),
#             Swish(),
#             nn.Linear(nh, nh),
#             Swish(),
#             nn.Linear(nh, nh),
#             Swish()
#         )
#         self.out = nn.Linear(nh, 1)

#     def forward(self, input1, input2):
#         h1 = self.net(input1)
#         h2 = self.net(input2)
#         return self.out((h1+h2).squeeze()).squeeze()
       

class DAE(nn.Module):

    def __init__(self,
                 latent_dim: int,
                 n_channels: int,
                 max_p: int = 2) -> None:
        super(DAE, self).__init__()
        # percent % of pixels to flip stochastically
        # for "noising" the input
        self.maxp = max_p
        
        self.latent_dim = latent_dim

        proj = [nn.Conv2d(1, n_channels, 3, 1, 1)]
        downsample = [
            BasicBlock(n_channels, n_channels, 2, norm=True),
            BasicBlock(n_channels, n_channels, 2, norm=True)
        ]
        main = [BasicBlock(n_channels, n_channels, 1, norm=True) for _ in range(1)]
        all = proj + downsample + main
        self.encoder = nn.Sequential(*all)
        self.fc = nn.Linear(n_channels * (28 //4)**2, latent_dim)


        # Build Decoder        
        proj = [nn.Linear(latent_dim, n_channels * (28 //4)**2),
                           nn.Unflatten(-1, (n_channels, 28//4, 28//4))]
        upsample = [
            BasicBlock(n_channels, n_channels, -2, norm=True),
            BasicBlock(n_channels, n_channels, -2, norm=True)
        ]
        main = [BasicBlock(n_channels, n_channels, 1, norm=True) for _ in range(1)]
        modules = proj + upsample + main
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Conv2d(n_channels, out_channels=1, kernel_size=1, padding=0)
                
        
    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        input = input.view(input.size(0), 1, 28, 28)
        result = self.encoder(input)
        result = result.view(result.size(0), -1)
        lc = self.fc(result)
        return lc

    
    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    
    def corrupt(self, x):
        """
        x is a one-hot image [batch_size, 1, 28, 28]
        """
        p = torch.randint(0,self.maxp+1,(1,))
        # randomly flip p% of the pixels
        flip = torch.bernoulli( (p.to(x.device) / 100.) * torch.ones_like(x).to(x.device) )
        x = (1 - x) * flip + (x * (1 - flip))
        return x
        
        
    def forward(self, noised_input, clean_input, **kwargs):
        z = self.encode(noised_input)
        return  [self.decode(z), clean_input, z]

    
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the DAE loss function.
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        z = args[2]        
        recons_loss = F.binary_cross_entropy_with_logits(recons.view(input.size(0), 28*28),
                                                         input, reduction='none')
        recons_loss = recons_loss.sum(-1).mean()                                                 
        return {'loss': recons_loss}

    
    def reconstruct(self, noised_input, clean_input):
        with torch.no_grad():
            outs = self.forward(noised_input, clean_input)
            return torch.bernoulli(F.sigmoid(outs[0]))
    
    def log_prob(self, input):
        input = input.view(input.size(0), 1, 28, 28)
        logits = self.decode(self.encode(input))
        log_prob = -F.binary_cross_entropy_with_logits(logits.view(input.size(0), 28*28),
                                                   input.view(input.size(0), 28*28),
                                                   reduction='none')
        return log_prob.sum(-1)



class Transformer(nn.Module):
    def __init__(self, model_name, protein):
        super().__init__()
        print('loading transformer weights from checkpoint...')
        if model_name == 'transformer-S':
            self.esm2, self.esm2_alphabet = pretrained.esm2_t12_35M_UR50D()
        elif model_name == 'transformer-M' or model_name == 'transformer':
            self.esm2, self.esm2_alphabet = pretrained.esm2_t30_150M_UR50D()
        elif model_name == 'transformer-L':
            self.esm2, self.esm2_alphabet = pretrained.esm2_t33_650M_UR50D()

        self.batch_converter = self.esm2_alphabet.get_batch_converter()
        
        self.wtseqs, wtids = io_utils.read_fasta(os.path.join(protein,
            'wt.fasta'), return_ids=True)
        
        _, _, self.wt_onehot = self.batch_converter([('wt', self.wtseqs[0])])
        self.wt_onehot = self.wt_onehot[:,1:-1]
        self.wt_length = self.wt_onehot.size()[0]
        self.wt_score = self.local_score(self.wt_onehot)
        self.potts_to_esm_perm = self.get_potts_to_esm_perm()
    
    def get_potts_to_esm_perm(self):
        '''
        returns a permutation of shape |potts vocab| x |esm vocab| mapping
        one-hot Potts vectors to one-hot ESM vectors
        '''
        potts_int_to_aa = data_utils.get_int_to_aa()
        esm_aa_to_int = self.esm2_alphabet.tok_to_idx
        perm = torch.zeros(len(potts_int_to_aa)-2,len(esm_aa_to_int))

        for k in range(len(potts_int_to_aa)):
            if not (potts_int_to_aa[k] == 'start' or  potts_int_to_aa[k] == 'stop'):
                perm[k, esm_aa_to_int[potts_int_to_aa[k]]] = 1
        return perm

    def translate_potts_to_esm(self, x):
        '''
        x is an N*L*q1 tensor of one-hot encodings encoded using the Potts encoding
        returns an N*L*q2 tensor of one-hot encodings using ESM encoding
        '''
        return x @ self.potts_to_esm_perm.to(x.device)
    
    def preprocess_onehot(self, x):
        '''
        not sure if I need to take a subset of x, deal with later
        '''
        return self.translate_potts_to_esm(x)
    
    def local_score(self, x):
        '''
        score that doesn't depend on relation to WT
        kind of like PLL or WT marginal except we condition on the current x, not WT.
        score = \sum_{i=1}^L log(logits[i]*x[i]) = tr log [(logits)^T x],
           where logits are from esm1b(x).
        Q: is tr log thing good because pytorch, or bad because we don't need to 
           calculate the entire matrix (logits)^T X?
        '''
        with torch.cuda.amp.autocast():
            logits = self.esm2(x)['logits']
            score = (x * torch.nn.functional.log_softmax(logits, -1)).sum(dim=[1,2])
            return score
    
    def forward(self, x, delta=True):
        s = self.local_score(x)
        if delta:
            return s - self.wt_score.to(x.device)
        else:
            return s

#########################

class PottsModel(nn.Module):
    def __init__(self, protein):
        super().__init__()
        potts_params = pkl.load(open(os.path.join(protein, 'potts.pkl'), 'rb'))
        self.J = nn.Parameter(torch.from_numpy(potts_params['J_ij']).float(), requires_grad=True)
        self.bias = nn.Parameter(torch.from_numpy(potts_params['h_i']).float(), requires_grad=True)
        self.index_list = potts_params['index_list']
        self.reg_coef = potts_params['reg_coef']
        self.seq_len = self.index_list.shape[0]
        self.n_tokens = 20
        self.data_dim = self.seq_len * self.n_tokens
        self.wtseqs, wtids = io_utils.read_fasta(os.path.join(protein,
            'wt.fasta'), return_ids=True)
        if '/' in wtids[0]:
            self.offset = int(wtids[0].split('/')[-1].split('-')[0])
        else:
            self.offset = 1
        self.index_list -= self.offset
        self.wt_H = self.hamiltonian(self.seqs2onehot(self.wtseqs))

        #print(self.wt_H)

    def seqs2onehot(self, seqs):
        """
        Convert a string of aa's to one-hot using offsets
        """
        seqs = [ seq[self.index_list[0]:self.index_list[-1]+1] for seq in seqs ]
        return torch.from_numpy(data_utils.seqs_to_onehot(seqs)).float()

    def preprocess_onehot(self, x):
        """
        Input:
            x is torch.FloatTensor of shape [batch_size, seq_len, vocab_size]
        Returns:
            subset of the onehot-encoded sequence given by offsets
        """
        return x[:,self.index_list[0]:self.index_list[-1]+1]

    def hamiltonian(self, x):
        
        """ Hamiltonian for x as torch.FloatTensor of shape [bs,seq_len,vocab_size] """
        x = x.reshape(x.size(0), self.seq_len, self.n_tokens)
        #assert list(x.size()[1:]) == [self.seq_len, self.n_tokens]
        Jx = torch.einsum("ijkl,bjl->bik", self.J, x)
        xJx = torch.einsum("aik,aik->a", Jx, x)  / 2  # J_ij == J_ji. J_ii is 0s.
        bias = (self.bias[None] * x).sum(-1).sum(-1)
        return xJx + bias

    def forward(self, x, delta=True):
        """
        x is one-hot (either [batch_size, seq_len*20] or [batch_size,seq_len,20])
        """
        if delta:
            return self.hamiltonian(x) - self.wt_H.to(x.device)
        else:
            return self.hamiltonian(x)    


class PottsTransformer(nn.Module):
    def __init__(self, protein):
        super().__init__()
        self.potts = PottsModel(protein)
        self.transformer = Transformer('transformer-M', protein)

    def preprocess_onehot(self, x):
        return (self.potts.preprocess_onehot(x), self.transformer.preprocess_onehot(x))

    def forward(self, x, delta=True):
        return self.potts(x[0], delta) + self.transformer(x[1], delta)


class AugmentedLinearRegression(nn.Module):
    def __init__(self, protein):
        super().__init__()
        self.potts = PottsModel(protein)
        self.potts.eval()
        self.coef_ = nn.ParameterList()
        self.intercept_ = nn.ParameterList()
        self.reg_coef = []
        
        for seed in range(20):
            linear_params = pkl.load(open(
                os.path.join(protein, f'results-predictor=ev+onehot-train=-1-seed={seed}-linear.pkl'), 'rb'))
            self.coef_ += [nn.Parameter(torch.from_numpy(linear_params['coef_']).float(), requires_grad=False)]
            self.intercept_ += [nn.Parameter(torch.FloatTensor([linear_params['intercept_']]), requires_grad=False)]
            self.reg_coef += [linear_params['reg_coef']]


    def forward(self, x): 
        """
        x are batched flattened one-hot proteins [batch_size, max_seq_len*vocabsize]
        delta_hamiltonian is [batch_size]

         concat([ sqrt(1 / reg_coef) * evo_score , sqrt(1 / reg_coef) * x ]
        """
        delta_hamiltonian = self.potts(self.potts.preprocess_onehot(x), delta=True)
        hamil_coef = self.potts.reg_coef
        # [batch_size, 1 + (max_seq_len * vocab_size)]
        x = x.view(x.shape[0], -1)
        y = []
        for W,b,r in zip(self.coef_, self.intercept_, self.reg_coef):
            x_i = torch.cat((math.sqrt(1 / hamil_coef) * delta_hamiltonian[...,None], math.sqrt(1 / r) * x ), 1)            
            y += [ (W * x_i).sum(1) + b ]
        return torch.stack(y,0).mean(0)  # [batch]


class OnehotCNN(nn.Module):
    def __init__(self, n_tokens, kernel_size, input_size, dropout=0.0):
        super().__init__()
        self.encoder = nn.Conv1d(n_tokens, input_size, kernel_size=kernel_size)
        self.embedding = nn.Sequential(
            nn.Linear(input_size, input_size*2),
            nn.ReLU(True)
        )
        self.decoder = nn.Linear(input_size*2, 1)
        self.n_tokens = n_tokens
        self.dropout = nn.Dropout(dropout)
        self.input_size = input_size
    
    def forward(self, x):
        """
        x is [batch_size, L, 20]
        """
        # encoder
        x = F.relu(self.encoder(x.transpose(1,2)).transpose(1,2))
        # embed
        x = self.embedding(x)
        # length-dim pool
        x  = torch.max(x, dim=1)[0]
        x = self.dropout(x)
        # decoder
        output = self.decoder(x)
        return output


class EnsembleMNIST:
    """
    Wraps around N PyTorch surrogates. implements 
    an API for getting predictions.
    """
    def __init__(self, weights_list, ensemble_class, device='cuda'):
        self.surrogates = []
        for i in range(len(weights_list)):
            self.surrogates += [ensemble_class()]
            self.surrogates[-1] = self.surrogates[-1].to(device)
            self.surrogates[-1].load_state_dict(torch.load(weights_list[i],
                                                map_location=torch.device(device))['model'])
            print(f'loaded {weights_list[i]}')
        
    
    def eval(self):
        for o in self.surrogates:
            o.eval()
            
    def train(self):
        for o in self.surrogates:
            o.train()

            
    def __call__(self, x1, x2):
        preds = []

        for o in self.surrogates:
            preds += [o(x1, x2)]
        preds = torch.stack(preds,0)
        return torch.mean(preds,0)


class EnsembleProtein:
    """
    Wraps around N PyTorch surrogates. implements 
    an API for getting predictions.
    """
    def __init__(self, weights_list, ensemble_class, device='cuda'):
        
        self.surrogates = []
        for i in range(len(weights_list)):
            self.surrogates += [ensemble_class()]
            self.surrogates[-1] = self.surrogates[-1].to(device)
            self.surrogates[-1].load_state_dict(torch.load(weights_list[i])['model'])
            print(f'loaded {weights_list[i]}')
        
    def eval(self):
        for o in self.surrogates:
            o.eval()
            
    def train(self):
        for o in self.surrogates:
            o.train()

    def __call__(self, x):
        """
        Assume x is a torch.FloatTensor of shape [batch_size, sequence_length, vocab_size]
        """
        preds = []
        for o in self.surrogates:
            preds += [o(x)]
        preds = torch.stack(preds,0)  # [ensemble_size,batch_size,1]
        return torch.mean(preds,0).squeeze()  # [batch_size]
