import pickle
import argparse
import os
import models
import helper_layers

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from helper import EarlyStopping
from helper import CustomMNIST
from tensorboardX import SummaryWriter # allows tracking and visualizing metrics such as loss and accuracy



def train(args, model, device, train_loader, val_loader, optimizer, epoch, writer):
    train_losses = [] # contains training losses over training batches
    model.train() # model in the training mode
    correct_train = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # necessary for general dataset: broadcast input
        data, _ = torch.broadcast_tensors(data, torch.zeros((helper_layers.steps,) + data.shape)) # adds time dimension to the first axis
        data = data.permute(1, 2, 3, 4, 0) # moves time dimension to the last axis

        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct_train += pred.eq(target.view_as(pred)).sum().item()

        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
            
        if batch_idx % args.log_interval == 0:
            print('Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data / helper_layers.steps), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            
            writer.add_scalar('Training Loss / Batch Index', loss, batch_idx + len(train_loader) * epoch)
    
    train_loss = sum(train_losses) / len(train_losses)
    train_acc = 100. * correct_train / len(train_loader.dataset)
    writer.add_scalar('Training Loss / Epoch', train_loss, epoch)
    writer.add_scalar('Training Accuracy / Epoch', train_acc, epoch)

    val_loss = 0.
    correct_val = 0
    model.eval() # model turns on the evaluation mode
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            data, _ = torch.broadcast_tensors(data, torch.zeros((helper_layers.steps,) + data.shape)) # adds time dimension to the first axis 
            data = data.permute(1,2,3,4,0) # moves time dimension to the last axis
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum')
            val_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct_val += pred.eq(target.view_as(pred)).sum().item()
            # val_losses.append(loss.item())
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100. * correct_val / len(val_loader.dataset)
        writer.add_scalar('Validation Loss / Epoch', val_loss, epoch)
        writer.add_scalar('Validation Accuracy / Epoch', val_acc, epoch)


    for i, (name, param) in enumerate(model.named_parameters()):
        if '_s' in name:
            writer.add_histogram(name, param, epoch)

    return model, train_loss, val_loss, train_acc, val_acc


def test(model, device, test_loader):
    model.eval()
    test_loss = 0.
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            data, _ = torch.broadcast_tensors(data, torch.zeros((helper_layers.steps,) + data.shape)) # adds time dimension to the first axis
            data = data.permute(1, 2, 3, 4, 0) # moves time dimension to the last axis

            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum') # sum up batch loss, target passed through softmax function
            test_loss += loss.item()  
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)

    return test_loss, correct


    

    
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    if (epoch % 10 == 0) and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


def main():
    # Arguments settings
    parser = argparse.ArgumentParser(description='Direct training for Deep Spiking Neural Networks')
    parser.add_argument('--train_batch_size', type=int, default=32, metavar='N',
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--val_batch_size', type=int, default=32, metavar='N',
                        help='Input batch size for validation (default: 32)')
    parser.add_argument('--test_batch_size', type=int, default=200, metavar='N',
                        help='Input batch size for testing (default: 200)')
    parser.add_argument('--epochs', type=int, default=800, metavar='N',
                        help='The number of epochs to train (default: 800)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training')
    parser.add_argument('--no_mps', action='store_true', default=False,
                        help='Disables MPS training (Macbook M1 GPU)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='The number of batches to wait before logging training status')

    args = parser.parse_args()

    torch.manual_seed(args.seed) # helps reproduce random results

    # Training settings
    # Pick the best device to run
    if not args.no_cuda and torch.cuda.is_available():
        device = 'cuda'
        print('Current device is {}'.format(device))
    elif not args.no_mps and torch.backends.mps.is_available():
        device = 'mps'
        print('Current device is {}'.format(device))
    else:
        device = 'cpu'
        print('Current device is {}'.format(device))


    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    writer = SummaryWriter() # Writer will output to ./runs/ directory by default.

    mnist_train_path = './datasets/MNIST/raw/mnist_train.csv'
    train_loader = torch.utils.data.DataLoader(
            CustomMNIST(csv_file=mnist_train_path,
                         transform=transforms.Compose([
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomCrop(28, padding=4),
                             transforms.Normalize(0.1307, 0.3082)
                         ])), batch_size=args.train_batch_size, shuffle=True)
    
    
    mnist_val_path = './datasets/MNIST/raw/mnist_val.csv'
    val_loader = torch.utils.data.DataLoader(
            CustomMNIST(csv_file=mnist_val_path,
                         transform=transforms.Compose([
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomCrop(28, padding=4),
                             transforms.Normalize(0.1301, 0.3074)
                         ])), batch_size=args.val_batch_size, shuffle=True)   
    

    mnist_test_path = './datasets/MNIST/raw/mnist_test.csv'
    test_loader = torch.utils.data.DataLoader(
            CustomMNIST(csv_file=mnist_test_path, 
                        transform=transforms.Compose([
                            transforms.Normalize(0.1325, 0.3105)
                        ])), batch_size=args.test_batch_size, shuffle=True)


    model = models.DSRN().to(device) # Deep Spiking Neural Network 19 layers model
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # checkpoint_path = './models/checkpoint.pt'
    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     model.load_state_dict(checkpoint)
    #     print('Model loaded.')
    
    early_stopper = EarlyStopping(verbose=True) # patience=7 by default
    for epoch in range(args.epochs):
        optimizer = adjust_learning_rate(optimizer, epoch) 
        print('Epoch {}: Current learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
        model, train_loss, val_loss, train_acc, val_acc = train(args, model, device, train_loader, val_loader, optimizer, epoch, writer)
        print('Average Training Loss: {}\t Average Validation Loss: {}'.format(train_loss, val_loss))
        print('Average Training Accuracy: {}\t Average Validation Accuracy: {}'.format(train_acc, val_acc))
        early_stopper(val_loss, model)

        if early_stopper.early_stop:
            break

    
    test_loss, correct = test(model, device, test_loader)
    print('\nTest Set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
        
    writer.flush()
    writer.close()

    



if __name__ == '__main__':
    main()
    
