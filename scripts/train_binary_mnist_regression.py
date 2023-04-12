from re import S
import torch
import torch.nn as nn
import argparse
import os
import numpy as np
from data.mnist import MNISTsumTo
from ppde.nets import MNISTRegressionNet
from ppde.nets import EnsembleMNIST
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)

        
def main(args):
    def my_print(s):
        print(s)
        #logger.write(str(s) + '\n')
        #logger.flush()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load data
    #train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    sumTo = args.sumTo
    train_data = MNISTsumTo(args, 'train', X=sumTo)
    val_data = MNISTsumTo(args, 'val', X=sumTo)
    test_data = MNISTsumTo(args, 'test', X=18 if args.make_eval_plots else sumTo)
    
    train_loader = torch.utils.data.DataLoader( train_data, shuffle=True, batch_size=args.batch_size )
    val_loader = torch.utils.data.DataLoader( val_data, shuffle=False, batch_size=args.test_batch_size )
    test_loader = torch.utils.data.DataLoader( test_data, shuffle=False, batch_size=args.test_batch_size )
    
    if not args.make_eval_plots:
        
        model = MNISTRegressionNet(64 if sumTo==18 else 16)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        
        # move to cuda
        model=model.to(args.device)
       

        if args.ckpt_path != '':
            d = torch.load(args.ckpt_path)
            model.load_state_dict(d['model'])
            optimizer.load_state_dict(d['optimizer'])          
            
        my_print(args.device)
        my_print(model)

        itr = 0
        running_loss = 0
        while itr <= args.n_iters:
            for batch in train_loader:
                x1,x2,y = batch
                x1=x1.to(args.device)
                x2=x2.to(args.device)
                y=y.to(args.device)
                
                optimizer.zero_grad()
                
                pred = model(x1, x2)
                    
                loss = torch.nn.functional.mse_loss(pred, y)
                loss.backward()
                
                optimizer.step()
                
                running_loss += loss.item()
                
                if itr % args.print_every == 0:
                    my_print("({}) loss = {:.6f}".format(itr, running_loss / args.print_every))
                    running_loss = 0.0
                
                if itr % args.save_every == 0:
                    d = {}
                    d['model'] = model.state_dict()
                    d['optimizer'] = optimizer.state_dict()
                    
                    if args.ensemble_id != -1:
                        save_name = f"{args.save_dir}/{args.model_type}_ensemble_{args.ensemble_id}_ckpt_{itr}.pt"
                    elif sumTo == 18:
                        save_name = f"{args.save_dir}/{args.model_type}_GT_ckpt_{itr}.pt"
                    else:
                        save_name = f"{args.save_dir}/{args.model_type}_ckpt_{itr}.pt"
                    torch.save(d, save_name)
                    
                if itr % args.eval_every == 0:               
                    model.eval()
                    
                    accuracy = []
                    with torch.no_grad():
                        for batch in val_loader:
                            x1,x2,y = batch
                            x1=x1.to(args.device)
                            x2=x2.to(args.device)
                            y=y.to(args.device)
                            
                            pred = model(x1, x2)

                            accuracy += [(y == torch.round(pred)).cpu().numpy()]
                    accuracy = np.concatenate(accuracy)
                    accuracy = accuracy.sum() / len(accuracy)
                    print(f'(Val {itr}) Accuracy {accuracy * 100} %')
                    
                    accuracy = []
                    with torch.no_grad():
                        for batch in test_loader:
                            x1,x2,y = batch
                            x1=x1.to(args.device)
                            x2=x2.to(args.device)
                            y=y.to(args.device)
                                
                            pred = model(x1, x2)
                            
                            accuracy += [(y == torch.round(pred)).cpu().numpy()]
                    accuracy = np.concatenate(accuracy)
                    accuracy = accuracy.sum() / len(accuracy)
                    print(f'(Held-out sums > 10 eval {itr}) OOD Accuracy {accuracy * 100} %')
                    
                    model.train()
                    
                itr += 1
        d = {}
        d['model'] = model.state_dict()
        d['optimizer'] = optimizer.state_dict()
        if args.ensemble_id != -1:
            save_name = f"{args.save_dir}/{args.model_type}_ensemble_{args.ensemble_id}_ckpt_{itr}.pt"
        else:
            save_name = f"{args.save_dir}/{args.model_type}_ckpt_{itr}.pt"
        torch.save(d, save_name)
    # make eval plots
    else:
        if args.sumTo == 18:
            model = MNISTRegressionNet(64 if sumTo==18 else 16)
            d = torch.load(args.ckpt_path)
            model.load_state_dict(d['model'])
            model = model.to(args.device)
        else:
            model = EnsembleMNIST( [f'./weights/mnist_models/ensemble_{i}_ckpt_25000.pt' for i in range(3)],
                                    lambda x=16: MNISTRegressionNet(x), args.device )
        model.eval()
                 
        my_print(args.device)
        my_print(model)

        all_test_preds = []
        all_test_ys = []

        with torch.no_grad():

            accuracy = []
            for batch in test_loader:
                x1,x2,y = batch
                x1=x1.to(args.device)
                x2=x2.to(args.device)
                y=y.to(args.device)
                

                pred = model(x1, x2)
                
                all_test_preds += [ torch.round(pred).view(-1).cpu().numpy() ]
                all_test_ys += [ y.view(-1).cpu().numpy() ]
                accuracy += [(y == torch.round(pred)).view(-1).cpu().numpy()]
            accuracy = np.concatenate(accuracy)
            accuracy = accuracy.sum() / len(accuracy)

        #all_val_preds = np.concatenate(all_val_preds)
        #all_val_ys = np.concatenate(all_val_ys)
        all_test_preds = np.concatenate(all_test_preds)
        all_test_ys = np.concatenate(all_test_ys)
        #all_val_preds = all_val_preds[all_val_ys <= 10]
        #all_val_ys = all_val_ys[all_val_ys <= 10]
        
        #all_test_preds = all_test_preds[all_test_ys > 10]
        #all_test_ys = all_test_ys[all_test_ys > 10]
        

        plt.figure()
        #plt.scatter(all_val_preds + np.random.normal(0,0.04,size=all_val_preds.size), all_val_ys + np.random.normal(0,0.04,size=all_val_ys.size), s=5)
        plt.scatter(all_test_preds + np.random.normal(0,0.05, size=all_test_preds.size), all_test_ys + np.random.normal(0,0.05,size=all_test_ys.size), s=5, label=f'{100*accuracy:.1f} %')
        
        plt.xticks([i for i in range(19)], [i for i in range(19)])
        plt.yticks([i for i in range(19)], [i for i in range(19)])
        plt.xlabel('pred. sum')
        plt.ylabel('actual sum')

        plt.plot([0, 18], [10.5, 10.5], c='r', linewidth=2)
        plt.legend()
        plt.tight_layout()
        method = 'oracle' if args.sumTo == 18 else 'surrogate'

        plt.savefig(args.results_path + f'/{method}_{args.model_type}_regression_eval.pdf')
        plt.savefig(args.results_path + f'/{method}_{args.model_type}_regression_eval.png')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--save_dir', type=str, default="./weights/mnist_models")
    parser.add_argument('--sumTo', type=int, default=10, 
                        help='10 for supervised experts, 18 for the oracle')
    parser.add_argument('--ensemble_id', type=int, default=0)
    parser.add_argument('--data_path', type=str, default="/ccs/proj/bie108/pemami")
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--device', type=str, default='cuda')
    
    # training
    parser.add_argument('--n_iters', type=int, default=40000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--print_every', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=.001)
    # eval
    parser.add_argument('--make_eval_plots', action='store_true')
    parser.add_argument('--results_path', type=str, default='./experiments/mnist/results')
    parser.add_argument('--flip_maxp', type=int, default=0)

    args = parser.parse_args()
    args.flip_maxp = 0
    main(args)
