import argparse
import os
import numpy as np
from src.metrics import proteins_transformer_score
import glob
import torch

general_args = argparse.ArgumentParser()
general_args.add_argument('--data_dir', type=str, default='/gpfs/alpine/bie108/proj-shared/pppo')
general_args.add_argument('--device', type=str, default='cuda')
general_args.add_argument('--results_dir', type=str, default='/gpfs/alpine/bie108/proj-shared/pppo/results/no_max_nmut')
general_args.add_argument('--sampler', type=str, default='')
general_args.add_argument('--prior', type=str, default='potts')
general_args.add_argument('--hub_dir', type=str, default='/gpfs/alpine/bie108/proj-shared/torch/hub/')
general_args.add_argument('--seed', type=int, default=1234567)

args = general_args.parse_args()

proteins = ["PABP_YEAST_Fields2013", "UBE4B_MOUSE_Klevit2013-nscor_log2_ratio", "GFP_AEQVI_Sarkisyan2016"]
MSAs = ["PABP_YEAST.a2m", "UBE4B_MOUSE.a2m", "GFP_AEQVI.a2m"]

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.hub.set_dir(args.hub_dir)

for msa,p in zip(MSAs,proteins):  
    runs = glob.glob(os.path.join(args.results_dir, p, f'{args.sampler}_prior={args.prior}*'))

    for r in runs:
        try:

            best_samples = np.load(os.path.join(r, 'population.npy'))
        except:
            continue
        
        print(f'found {r} redoing MSA transformer scores')

        best_samples = torch.from_numpy(best_samples).to(args.device)

        transformer_scores = proteins_transformer_score(best_samples, os.path.join(args.data_dir, 'weights', p), 
                                os.path.join(args.data_dir, 'alignments', msa), 500)


        np.save(os.path.join(r, 'transformer_scores.npy'), transformer_scores)

print('done')

