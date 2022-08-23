# Bert deep learning model  
# BERT - Bidirectional Encoder Representations from Transformers
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import pickle
import numpy as np
from torch.utils.data import random_split 
import copy  
from tqdm import tqdm
import datetime
from text_loader_bert import Textloader


def split_dataset(dataset, train_percentage, validation_percentage):
    
    """This function splits the dataset into train, validation, and test and uses dataloader to store the values as iterables
    for all three datasets.

    Args:
        dataset (Dataset): The dataset of features and labels as tuples and torch tensor format
        train_percentage (float): The training percentage of the dataset
        validation_percentage (float): The validation percentage of the dataset
    
    Returns:
        Tuple: Tuple containing train, validation, and test data in form of dataloader objects.
    """
    train_validation = int(len(dataset) * train_percentage)
    validation_split = int(train_validation * validation_percentage)
    test_split = int(len(dataset) - train_validation)

    train_data, test_data = random_split(dataset, [train_validation, test_split], generator=torch.Generator().manual_seed(100))
    train, validation = random_split(train_data, [int(train_validation - validation_split), validation_split], generator=torch.Generator().manual_seed(100))

    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    valid_loader = DataLoader(validation, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    
    for file_name, dataloader in [('text_train_data.pkl', train_loader), ('text_test_data.pkl', test_loader), ('text_validation_data.pkl', valid_loader)]:
        with open(file_name, 'wb') as file:
            pickle.dump(dataloader, file)

    return train_loader, valid_loader, test_loader


# Early stopping
class EarlyStopping():
    """This class implements regularization during training using early stopping to prevent overfitting

    Args:
        patience (int): Number of times validation loss is allowed to increase in a row every epoch or so.
    """
    def __init__(self, patience=4):

        self.patience = patience
        self.counter = 0
        self.early_stop = False

    def __call__(self, previous_val_loss, curr_val_loss):
        """This magic method compares validation losses and adds to counter if loss increases, else resets the counter
        everytime an instance of the class is called
        
        Args:
            previous_val_loss (float): The validation loss in previous epoch
            curr_val_loss (float): The current validation loss
        """
        if curr_val_loss >= previous_val_loss:
            self.counter +=1
            if self.counter >= self.patience:  
                self.early_stop = True
        else:
            self.counter = 0


# Testing our model to calculate accuracy
def test_accuracy(test_loader, device, model):
    """This function loops through the test data, calculates predictions, locates classes with max probability,
    compares the predictions with the labels and calculates the accuracy.

    Args:
        model (Model): CNN model 
        device (device): Whether we are working on GPU or CPU
        test_loader (DataLoader): Test data

    Returns:
        float: The accuracy of model predictions 
    """
    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in tqdm(test_loader):
            text, labels = data
            text, labels = text.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(text)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            # print(torch.max(outputs.data, 1))
            total += labels.size(0)
            correct += torch.sum(predicted == labels.data)
    
    accuracy = torch.div(100 * correct.double(), total)
    print(f'Got {correct} / {total}')
    # print('Accuracy of the network on the test images: {} %'.format(accuracy))
    return accuracy


class Classifier(torch.nn.Module):
    """This CNN class uses the torch Sequential module to build layers of the text data neural network which includes 
    convolutional layers, dropout layers, max pooling layers, linear layers with ReLU as activation functions, with
    the last layer having 13 outputs for each category class in the dataset.
    
    Args:
        ngpu (int): The number of CPU cores to use for training
        num_classes (int): The number of classes to predict by the trained model
        input_size (int): The input dimension fed into first Convolutional layer
    
    """

    def __init__(self, ngpu, num_classes, input_size=768):
        super(Classifier, self).__init__()
        self.ngpu = ngpu
        self.main = torch.nn.Sequential(torch.nn.Conv1d(input_size, 256, kernel_size=3, stride=1, padding=1),
                                  torch.nn.ReLU(),
                                  torch.nn.MaxPool1d(kernel_size=2, stride=2),
                                  torch.nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
                                  torch.nn.ReLU(),
                                  torch.nn.MaxPool1d(kernel_size=2, stride=2),
                                  torch.nn.Dropout(p=0.2),
                                  torch.nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
                                  torch.nn.Dropout(p=0.2),
                                  torch.nn.ReLU(),
                                  torch.nn.MaxPool1d(kernel_size=2, stride=2),
                                  torch.nn.Dropout(p=0.2),
                                  torch.nn.ReLU(),
                                  torch.nn.Flatten(),
                                  torch.nn.Linear(128 , 64),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(64, num_classes))
    def forward(self, input):
        """Returns prediction on the features using the defined neural network"""
        x = self.main(input)
        return x


def validation(model, device, valid_loader, loss_function):
    """This function uses the CNN model to evaluate the loss on the validation data every certain epochs

    Args:
        model (Model): CNN model 
        device (device): Whether we are working on GPU or CPU
        valid_loader (DataLoader): Validation data
        loss_function (torch.nn.Module): To evaluate loss of our predictions

    Returns:
        float: The average validation loss every epoch
    """
    hist_val_acc = []
    model.eval() # it tells your model that you are testing the model
    loss_total = 0
    hist_val_loss = []

    # Test validation data
    with torch.no_grad(): # Deactivate autograd, requires_grad is set to False, do not calculate gradients of new variables as we testing only
        print('\n')
        progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for _, (features, labels) in progress_bar:
            features, labels = features.to(device), labels.to(device)

            output = model(features)
            loss = loss_function(output, labels)
            accuracy = (torch.sum(torch.argmax(output, dim=1) == labels).item()) / len(labels)
            hist_val_acc.append(accuracy)
            hist_val_loss.append(loss.item())
            progress_bar.set_description(f"Validation metrics: acc = {round(float(accuracy), 2)}. mean_val_acc = {round(np.mean(hist_val_acc), 2)}. mean_val_loss = {round(np.mean(hist_val_loss), 2)}")
            loss_total += loss.item()

    return loss_total / len(valid_loader), round(np.mean(hist_val_acc), 2)
  

def train(model, device, train_loader, valid_loader, epochs=10):

    """This function trains the CNN model, loops through the training data for a set number of epochs, 
    calculates predictions and loss, updates gradients and model parameters, prints loss in Tensorboard, saves the model every few epochs,
    and returns the trained CNN model.

    Args:
        model (Model): CNN model 
        device (device): Whether we are working on GPU or CPU
        train_loader (DataLoader): Training data
        valid_loader (DataLoader): Validation data
        epochs (int): Number of times we loop through training data to improve our model parameters

    Returns:    
        model (Model): Trained CNN model
    """
     # Early stopping
    early_stopping = EarlyStopping(patience=4)
    last_loss = np.inf
    running_corrects = 0
            

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=3e-5)
    writer = SummaryWriter()
    batch_idx = 0
    total = 0 
    prev_val_acc = 0
    

    for epoch in range(epochs+1):
        print('\n')
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        hist_acc = [] 
        model.train()
        for _, (features, labels) in progress_bar:
            features, labels = features.to(device), labels.to(device)
            prediction = model(features)
            # zero the parameter gradients
            optimiser.zero_grad()

            accuracy = (torch.sum(torch.argmax(prediction, dim=1) == labels).item()) / len(labels)
            hist_acc.append(accuracy)

            loss = F.cross_entropy(prediction, labels)
            _, preds = torch.max(prediction, 1)
            loss.backward()

            # backward + optimize only if in training phase
            optimiser.step()
            writer.add_scalar('Training Loss', loss.item(), batch_idx)

            batch_idx += 1
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            progress_bar.set_description(f"Epoch = {epoch}/{epochs}. acc = {round(float(accuracy), 2)}. mean_train_acc = {round(np.mean(hist_acc), 2)}. Loss = {round(float(loss), 2)}")

        # Early stopping
        validation_loss_per_epoch, val_acc = validation(model, device, valid_loader, F.cross_entropy) 
        writer.add_scalar('Validation Loss', validation_loss_per_epoch, batch_idx)
        writer.add_scalar('Validation Accuracy', val_acc, batch_idx)
        early_stopping(last_loss, validation_loss_per_epoch)
        last_loss = validation_loss_per_epoch
            
        # Only save the best performing model (best accuracy on validation set)
        if val_acc > prev_val_acc:
            prev_val_acc = val_acc
            model.to(device)
            torch.save({'model_state_dict': copy.deepcopy(model.state_dict())}, 'text_model.pt')
                

        if early_stopping.early_stop:
            print("Early stopping invoked! We are at epoch:", epoch)
            # torch.save({'model_state_dict': copy.deepcopy(model.state_dict())}, 'text_model.pt')
            return model

    print('End of epochs reached with best model saved as text_model.pt')
    return model
        

if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Textloader()
    ngpu = 2
    num_classes = dataset.num_classes

    if os.path.exists('text_train_data.pkl'):
        train_loader = pd.read_pickle('text_train_data.pkl')
        test_loader = pd.read_pickle('text_test_data.pkl')
        valid_loader = pd.read_pickle('text_validation_data.pkl')

    else:
        dataset = Textloader()
        train_loader, valid_loader, test_loader = split_dataset(dataset, 0.8, 0.1)

    if os.path.exists('text_model.pt'):
        further_training = input('Do you want the model to be trained further: ')
        model = Classifier(ngpu=ngpu, num_classes=num_classes)
        checkpoint = torch.load('text_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        if further_training.lower() == 'yes':
            model_cnn = train(model, device, train_loader, valid_loader, epochs=10)

        model.eval()
        acc = test_accuracy(test_loader, device, model)
        print('Accuracy of the network on the test data (product descriptions): {} %'.format(acc))

    else:
        model = Classifier(ngpu=ngpu, num_classes=num_classes)
        model.to(device)
        model_cnn = train(model, device, train_loader, valid_loader, epochs=30)
        acc = test_accuracy(test_loader, device, model_cnn)
        print('Accuracy of the network on the test data (product descriptions): {} %'.format(acc))


