import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np
import os
import pandas as pd

from lesion_dataset import LesionDataset
from module_densenet_pytorch import DenseNetCNN

import doctest

import warnings

'''
HyperParams is an object containing
 - n_classes     Int (The number of classes being classified)
 - n_epochs      Int (The number of epochs during training)
 - batch_size    Int (The number of images in each batch)
 - learning_rate Num (The learning rate used by the optimizer)
interp. hyper parameters used by machine learning model during training

Example:
params = {
    n_classes: 1,
    n_epochs:  20,
    batch_size: 4,
    learning_rate = 0.003,
}
'''
def fn_for_hyper_params(hyperparams):
    ... hyperparams.n_classes
    ... hyperparams.n_epochs
    ... hyperparams.batch_size
    ... hyperparams.learning_rate


# =========
# Constants

hyperparams = {
    n_classes     = 1, # Binary Classification
    n_epochs      = 20,
    batch_size    = 4,
    learning_rate = 0.003
}


# =========
# Functions

def setup():
    """
    Call various world setup functions
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    # Use GPU
    if not "GeForce" in torch.cuda.get_device_name(
            torch.cuda.current_device()):
        print("Stopped. Not using GPU.")
        exit()
    else:
        print("Using: ", torch.cuda.get_device_name(
            torch.cuda.current_device()))

'''
=============
Main Function

Call with:
 - ...
'''

def main():
    setup()

if __name__ == "__main__":
    main()

# Ignore warnings for now


# Make sure we are using GPU


# Training Parameters
n_classes = 1 # Binary Classification
n_epochs = 20
batch_size = 4
learning_rate = 0.003

model = DenseNetCNN(n_classes)
model.cuda()

# Loss / Optim
# Attempt 1
# criterion = BCELoss()
# optimizer = Adam(model.parameters(), lr=learning_rate)

# Attempt 2
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


# Load Training Data
train_df = pd.read_csv('./data_csvs/train_only_ids.csv')
train_ds = LesionDataset(train_df, './lesion_images/all_images_processed_2/')
train_dl = DataLoader(train_ds, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

valid_df = pd.read_csv('./data_csvs/test_only_ids.csv')
valid_ds = LesionDataset(valid_df, './lesion_images/all_images_processed_2/')
valid_dl = DataLoader(valid_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

dataloaders = {
        'train': train_dl,
        'valid': valid_dl}

# Training and Validation
best_train_loss = np.Inf
best_train_acc = 0.0

last_train_loss = np.Inf
last_train_acc = 0.0

train_acc_list = []

best_valid_loss = np.Inf
best_valid_acc = 0.0

last_valid_loss = np.Inf
last_valid_acc = 0.0

valid_acc_list = []

train_loss_decreasing_count = 0
strike = 0

for epoch in range(1, n_epochs + 1):

    if epoch > 1:
        checkpoint = {
                'model': DenseNetCNN(n_classes),
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}

        torch.save(checkpoint, 'pytorch_baseline_checkpoint.pth')

    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_correct = 0
        epoch_loss = 0.0
        epoch_acc = 0.0
    
        for i, data in enumerate(dataloaders[phase], 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs.float())
            loss = criterion(outputs, torch.max(labels, 1)[1]) # For CrossEntropyLoss
            # loss = criterion(outputs, labels.float()) # For BCELoss

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            _, preds = torch.max(outputs, 1)
            preds = preds.reshape(labels.shape) # Ensure comparison is of the same size

            running_loss += loss.detach() * inputs.size(0)
            running_correct += torch.sum(preds.float() == labels.float())

            epoch_loss = running_loss.item() / ((i + 1) * batch_size)
            epoch_acc = running_correct.item() / ((i + 1) * batch_size) * 100

            if i % 20 == 19:
                os.system('cls' if os.name == 'nt' else 'clear')
                
                phase_progress = ((i + 1) * batch_size) / len(dataloaders[phase].dataset) * 100

                print("Current Phase: ", phase)
                print("Epoch: ", epoch, "/", n_epochs)
                print("Phase Progress: {:.3f} %".format(phase_progress))
                print("---")
                print("Running Loss: {:.3f}".format(epoch_loss))
                print("Running Acc: {:.3f} %".format(epoch_acc))
                print("---")
                print("Best Train Loss: {:.3f}".format(best_train_loss))
                print("Best Train Acc: {:.3f} %".format(best_train_acc))
                print("Last Train Loss: {:.3f}".format(last_train_loss))
                print("Last Train Acc: {:.3f} %".format(last_train_acc))
                print("---")
                print("Best Valid Loss: {:.3f}".format(best_valid_loss))
                print("Best Valid Acc: {:.3f} %".format(best_valid_acc))
                print("Last Valid Loss: {:.3f}".format(best_valid_loss))
                print("Last Valid Acc: {:.3f} %".format(best_valid_acc))

        if phase == 'train':
            
            last_train_loss = epoch_loss
            last_train_acc  = epoch_acc

            if last_train_loss < best_train_loss:
                best_train_loss = last_train_loss
                train_loss_decreasing_count += 1
            else:
                train_loss_decreasing_count = 0
            
            if last_train_acc > best_train_acc:
                best_train_acc = last_train_acc

        elif phase == 'valid':

            last_valid_loss = epoch_loss
            last_valid_acc  = epoch_acc

            if last_valid_loss < best_valid_loss:
                best_valid_loss  = last_valid_loss

            if last_valid_acc > best_valid_acc:
                best_valid_acc = last_valid_acc

    if epoch > 5 and train_loss_decreasing_count == 0:
        strike += 1
        if strike == 3:
            break

print('---')
print('Finished Training. Saving Model...')
torch.save(model.state_dict(), 'pytorch_densenet151_baseline_model.pt')

