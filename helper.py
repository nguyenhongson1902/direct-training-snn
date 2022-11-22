import numpy as np
import torch
from torch.utils.data import Dataset


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='./models/checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class CustomMNIST(Dataset):
    """
        Customized MNIST for loading from csv files. There's no need to use transform.ToTensor()
    """
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the MNIST csv file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mnist = np.loadtxt(csv_file, delimiter=',', dtype=np.float32)
        self.transform = transform
    
    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        """
            Args:
                idx (int): Index of image in the dataset.
        """
        label = torch.from_numpy(np.asarray(self.mnist[idx, 0])).long()
        image = torch.from_numpy(self.mnist[idx, 1:].reshape(1, 28, 28))
        image = image / 255.
        if self.transform:
            self.transform(image)
        return image, label

