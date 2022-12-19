import torch
from torchvision import datasets, transforms


class MNISTsumTo(torch.utils.data.Dataset):
    """
    Train/val on sums up to X, test on larger sums.
    """
    def __init__(self, args, mode, X=10):
        data = datasets.MNIST(args.data_path,
                     train=True if mode == 'train' or mode == 'val' else False,
                     download=True, transform=transforms.Compose([transforms.ToTensor()]))
        self.mode = mode
        self.maxp = args.flip_maxp
        self.train_size = 50000 if X==18 else 5000
        self.val_size = 10000
        
        self.data_map = []
        label_file = f'./data/MNISTsum{X}_' + mode + '.txt'
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip('\n').split(',')
                self.data_map += [(int(line[0]), int(line[1]), int(line[2]))]
        
        if mode == 'train' or 'val':
            x = data.train_data.float() / 255.
            self.x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
            self.y = data.train_labels.float()
            
        else:
            x = data.test_data.float() / 255.
            self.x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
            self.y = data.test_labels.float()
   
        if mode == 'train':
            self.x = self.x[0:50000]
            self.y = self.y[0:50000]
            self.size = self.train_size
        elif mode == 'val':
            self.x = self.x[50000:60000]
            self.y = self.y[50000:60000]
            torch.manual_seed(777)
            # binarize
            self.x = torch.bernoulli(self.x)
            self.size = self.val_size
        elif mode == 'test':
            torch.manual_seed(777)
            # binarize
            self.x = torch.bernoulli(self.x)
            self.size = self.val_size
                
    def __len__(self):
        return self.size
    
    
    def __getitem__(self, i):
        x1, x2, y = self.data_map[i]
            
        # dynamically binarize
        x1 = self.x[x1]
        x2 = self.x[x2]
        y = torch.FloatTensor([y])        
        if self.mode == 'train':
            x1 = torch.bernoulli(x1)
            x2 = torch.bernoulli(x2)
            p1 = torch.randint(0,self.maxp+1,(1,))
            p2 = torch.randint(0,self.maxp+1,(1,))
            
            # randomly flip p% of the pixels
            flip1 = torch.bernoulli( (p1 / 100.) * torch.ones_like(x1) )
            flip2 = torch.bernoulli( (p2 / 100.) * torch.ones_like(x2) )   
            x1 = (1 - x1) * flip1 + (x1 * (1 - flip1))
            x2 = (1 - x2) * flip2 + (x2 * (1 - flip2)) 
            
            # 50% swap order of x1/x2
            # if torch.rand(1) <= 0.5:
            #     temp = x1.clone()
            #     x1 = x2
            #     x2 = temp
            # add small label noise for smoothing
            y = torch.distributions.normal.Normal(y, 0.1).sample()
        
        return x1, x2, y.squeeze()
    
    
    
if __name__ == '__main__':
    import numpy as np
    # train/val
    # 10 for train/ 18 for GT
    sum_upto = 18

    data = datasets.MNIST('./temp', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    y = data.train_labels.int()

    y_train = y[0:50000]
    y_val = y[50000:60000]

    possible_pairs = [[] for _ in range(10)]
    for i in range(10):
        for j in range(10):
            if i + j <= sum_upto:
                possible_pairs[i] += [j]

    print(possible_pairs)
    
    # group labels 
    grouped_train_labels = [[] for _ in range(10)]
    grouped_val_labels = [[] for _ in range(10)]
    for idx in range(50000):
        grouped_train_labels[y_train[idx]] += [idx]
    for idx in range(10000):
        grouped_val_labels[y_val[idx]] += [idx]
        
    with open(f'MNISTsum{sum_upto}_train.txt', 'w+') as f:
        for idx in range(50000):
            label = y_train[idx]
            # get a random possible digit pair for this sum
            pairing = np.random.choice(possible_pairs[label])
            # get a random image for this digit
            digit_idx = np.random.choice(grouped_train_labels[pairing])
            
            f.write(f'{idx},{digit_idx},{label+pairing}\n')
    
    with open(f'MNISTsum{sum_upto}_val.txt', 'w+') as f:
        for idx in range(10000):
            label = y_val[idx]
            # get a random possible digit pair for this sum
            pairing = np.random.choice(possible_pairs[label])
            # get a random image for this digit
            digit_idx = np.random.choice(grouped_val_labels[pairing])
            
            f.write(f'{idx},{digit_idx},{label+pairing}\n')
            
    print('creating test set')
    # test on sums > 10
    data = datasets.MNIST('./temp', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    y = data.test_labels.int()
    
    possible_pairs = [[] for _ in range(10)]
    for i in range(10):
        for j in range(10):
            if sum_upto == 18:
                possible_pairs[i] += [j]
            elif i + j > sum_upto:
                possible_pairs[i] += [j]

    print(possible_pairs)
    
    grouped_labels = [[] for _ in range(10)]
    for i in range(y.shape[0]):
        grouped_labels[y[i]] += [i]
        
    with open(f'MNISTsum{sum_upto}_test.txt', 'w+') as f:
        for idx in range(y.shape[0]):
            label = y[idx]
            
            while len(possible_pairs[label]) == 0:
                label = np.random.choice(list(range(10)))
                idx = np.random.choice(grouped_labels[label])
            pairing = np.random.choice(possible_pairs[label])
            digit_idx = np.random.choice(grouped_labels[pairing])
            
            f.write(f'{idx},{digit_idx},{label+pairing}\n')
            