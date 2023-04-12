import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import torchvision
from ppde.nets import DAE
from ppde.third_party.grathwohl import vamp_utils


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
def main(args):
    
    makedirs(args.save_dir)
    logger = open("{}/log.txt".format(args.save_dir), 'w')
  
    def my_print(s):
        print(s)
        logger.write(str(s) + '\n')
        logger.flush()
        
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load data
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0),
                                                            args.input_size[0], args.input_size[1], args.input_size[2]),
                                                     p, normalize=True, nrow=int(x.size(0) ** .5)) 
    
    model = DAE(latent_dim=16, n_channels=64, max_p=args.flip_maxp)
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # move to cuda
    model=model.to(args.device)
    if args.ckpt_path is not None:
        d = torch.load(args.ckpt_path)
        model.load_state_dict(d['model'])
        optimizer.load_state_dict(d['optimizer'])
        itr = d['step']
        print(f'loaded {args.ckpt_path}')
    else:
        itr = 0
        
    model.train()
    
    my_print(args.device)
    my_print(model)
    
    running_bce = 0
    
    while itr <= args.n_iters:
        for x in train_loader:
            x = x[0].to(args.device)

            with torch.no_grad():
                x_noisy = model.corrupt(x)
            
            optimizer.zero_grad()
            outs = model(x_noisy, x)
              
            loss = model.loss_function(*outs)
            recon = loss['loss']

            recon.backward()
            
            optimizer.step()

            running_bce += recon.item()
            
            if itr > 1 and itr % args.print_every == 0:
                my_print("({}) bce = {:.6f}".format(
                    itr, running_bce / args.print_every)
                )
                running_bce = 0.0
                plot("{}/latest_recon.png".format(args.save_dir), model.reconstruct(x_noisy, x))
            
            if itr % args.eval_every == 0:
                d = {}
                d['model'] = model.state_dict()
                d['optimizer'] = optimizer.state_dict()
                d['step'] = itr
                
                save_name = f"{args.save_dir}/mnist_binary_dae_{itr}.pt"
                torch.save(d, save_name)
                model.to(args.device)
                
            itr += 1
            
if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--save_dir', type=str, default="./weights/mnist_models/mnist_binary_dae.pt")
    parser.add_argument('--dataset_name', type=str, default='static_mnist')
    parser.add_argument('--data_path', type=str, default="")
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--device', type=str, default='cuda')
    # training
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=500)
    parser.add_argument('--eval_every', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--flip_maxp', type=int, default=15)

    
                            
    args = parser.parse_args()
    main(args)
