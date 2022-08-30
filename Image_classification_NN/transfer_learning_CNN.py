from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import pickle
import numpy as np
from torchvision.models import ResNet50_Weights
from torchvision import models
from torch.utils.data import random_split 
import copy  
from tqdm import tqdm
import datetime
from image_loader import ImageDataset


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

    
    for file_name, dataloader in [('train_data.pkl', train_loader), ('test_data.pkl', test_loader), ('validation_data.pkl', valid_loader)]:
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
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            # print(torch.max(outputs.data, 1))
            total += labels.size(0)
            correct += torch.sum(predicted == labels.data)
    
    accuracy = torch.div(100 * correct.double(), total)

    # print('Accuracy of the network on the test images: {} %'.format(accuracy))
    return accuracy


class Classifier(torch.nn.Module):
    """This CNN class starts of with the resnet50 pretrained model where only the fully connected and layer 4 are unfreezed 
    for training. The last layer will be a linear layer using the output from the resnet50 as input and outputs the number of 
    product categories. The torch Sequential module is used to connect the resnet50 and linear layer.
    
    Args:
        ngpu (int): The number of CPU cores to use for training
        num_classes (int): The number of classes to predict by the trained model
    
    """

    def __init__(self, ngpu, num_classes):
        super(Classifier, self).__init__()
        self.ngpu = ngpu
        resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        for param in resnet50.parameters(): 
            param.requires_grad = False
        # unfreeze last layer params
        for param in resnet50.fc.parameters():
            param.requires_grad = True

        for param in resnet50.layer4.parameters():
            param.requires_grad = True

        out_features = resnet50.fc.out_features
        self.linear =torch.nn.Linear(out_features, num_classes).to(device)
        self.main = torch.nn.Sequential(resnet50, self.linear).to(device)

    def forward(self, input):
        """Returns prediction on the features using the defined neural network"""
        x = self.main(input)
        return x

# CNN model trained from scratch 
# class CNN(torch.nn.Module):
#     """This CNN class uses the torch Sequential module to build layers of the neural network which includes 
#     convolutional layers, dropout layers, max pooling layers, a linear layer with ReLU as activation functions, and
#     softmax being used at the end to output probabilities of each class in the dataset."""

#     def __init__(self):
#         super().__init__()
#         # Declaring the Architecture
#         self.layers = torch.nn.Sequential(
#             torch.nn.Conv2d(3, 200, 5, 2), # Kernel size 7 with stride 2
#             torch.nn.MaxPool2d(4,4),
#             torch.nn.ReLU(),
            
#             torch.nn.Conv2d(200, 100, 3),
#             # torch.nn.MaxPool2d(2, 2), # Max pooling (2, 2) filter
#             torch.nn.Dropout(p=0.2),
#             torch.nn.ReLU(),

#             torch.nn.Conv2d(100, 50, 3),
#             torch.nn.MaxPool2d(2, 2),
#             torch.nn.Dropout(p=0.3),
#             torch.nn.ReLU(),

#             torch.nn.Flatten(),
#             torch.nn.Linear(1250, 100),
#             torch.nn.Linear(100, 13), #  Predicting 13 product categories
#             torch.nn.Softmax(dim=1)
#         )

#     def forward(self, features):  
#         """Returns prediction on the features using the defined neural network""" 
#         return self.layers(features)

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
    early_stopping = EarlyStopping(patience=9)
    last_loss = np.inf
    running_corrects = 0
    # start = time.time()
    ct = datetime.datetime.now()
    timestamp = ct.ctime().split()[-2].replace(':', '-')
    # save weights at the end of every epoch
    eval_path = 'model_evaluation'
    model_filename = f'model_{timestamp}'
    model_path = f'{eval_path}/{model_filename}'
    final_models_path = 'final_models'
    paths = [eval_path, model_path, final_models_path]
    
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
            

    optimiser = torch.optim.Adam(model.parameters(), lr=3e-4)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=2, gamma=0.6)
    # weight decay adds penalty to loss function, shrinks weights during backpropagation to prevent overfitting and exploding gradients
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

        exp_lr_scheduler.step()
        # Early stopping
        validation_loss_per_epoch, val_acc = validation(model, device, valid_loader, F.cross_entropy) 
        writer.add_scalar('Validation Loss', validation_loss_per_epoch, batch_idx)
        writer.add_scalar('Validation Accuracy', val_acc, batch_idx)
        early_stopping(last_loss, validation_loss_per_epoch)
        last_loss = validation_loss_per_epoch

        
        # ct stores current time
        ct = datetime.datetime.now()
        timestamp = ct.ctime().split()[-2]

        # save weights at the end of every epoch
        # elapsed_time = time.time() - start
        weights_path = model_path + '/weights'
        
        if not os.path.exists(weights_path):
            os.mkdir(weights_path)

            
        # Only save the best performing model (best accuracy on validation set)
        if val_acc > prev_val_acc:
            save_model(epoch, model, optimiser, val_acc, loss, weights_path)
            prev_val_acc = val_acc
            model.to(device)
            torch.save({'model_state_dict': copy.deepcopy(model.state_dict())}, f'{final_models_path}/image_model.pt')
                

        if early_stopping.early_stop:
            print("Early stopping invoked! We are at epoch:", epoch)
            torch.save(model, 'CNN_model.pt')
            return model

    # torch.save(model.state_dict(), 'model_cnn')
    torch.save(model, 'CNN_model.pth') # save the final model as well for comparison 
    print('End of epochs reached with best model saved in final_models')
    return model
        
def save_model(epoch, model, optimiser, val_acc, loss, weights_path):
    """This function changes model to cpu mode, saves the model weights, epoch we reached, optimizer details,
    validation accuracy and the training loss so we can restore any model if we want to explore and compare results later on.

    Args:
        epoch (int): Number of epoch we reached at the end of training
        model (torchvision.models): Model our training function returned
        optimiser (torch.optim): Optimiser we used to train
        val_acc (float): Average validation accuracy per epoch at the end of training
        loss (float): Average training loss per epoch 
        weights_path (str): The path to the weights folder contained in the model evaluation directory

    """
    
    model.to('cpu')
    torch.save({
        'epoch': epoch,
        'model_state_dict': copy.deepcopy(model.state_dict()),
        'optimizer_state_dict': optimiser.state_dict(),
        'validation_acc': val_acc,
        'training+loss': loss}, os.path.join(weights_path, f'epoch_{epoch}_results'))



if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ImageDataset()
    ngpu = 2
    num_classes = dataset.num_classes

    if os.path.exists('train_data.pkl'):
        train_loader = pd.read_pickle('train_data.pkl')
        test_loader = pd.read_pickle('test_data.pkl')
        valid_loader = pd.read_pickle('validation_data.pkl')

    else:
        dataset = ImageDataset()
        train_loader, valid_loader, test_loader = split_dataset(dataset, 0.8, 0.1)

    if os.path.exists('final_models/image_model.pt'):
        model = Classifier(ngpu=ngpu, num_classes=num_classes)
        checkpoint = torch.load('final_models/image_model.pt')
        # model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        acc = test_accuracy(test_loader, device, model)
        print('Accuracy of the network on the test images: {} %'.format(acc))

    else:

        model = Classifier(ngpu=ngpu, num_classes=num_classes)
        model.to(device)
        model_cnn = train(model, device, train_loader, valid_loader, epochs=20)
        acc = test_accuracy(test_loader, device, model_cnn)
        print('Accuracy of the network on the test images: {} %'.format(acc))
