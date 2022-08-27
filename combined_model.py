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
from combined_dataloader import ImageTextDataset
import numpy as np
from torchvision.models import ResNet50_Weights
from torchvision import models
from torch.utils.data import random_split
import warnings
warnings.filterwarnings('ignore') 


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

    
    for file_name, dataloader in [('image_text_train_data.pkl', train_loader), ('image_text_test_data.pkl', test_loader), ('image_text_validation_data.pkl', valid_loader)]:
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
            images, text,  labels = data
            images, text, labels = images.to(device), text.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images, text)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            # print(torch.max(outputs.data, 1))
            total += labels.size(0)
            correct += torch.sum(predicted == labels.data)
    
    accuracy = torch.div(100 * correct.double(), total)
    print(f'Got {correct} / {total}')
    # print('Accuracy of the network on the test images: {} %'.format(accuracy))
    return accuracy


class TextClassifier(torch.nn.Module):
    """This CNN class uses the torch Sequential module to build layers of the text data neural network which includes 
    convolutional layers, dropout layers, max pooling layers, linear layers with ReLU as activation functions, with
    the last layer having 128 outputs which will be concatenated with the output from the ImageClassifier model.
    
    Args:
        ngpu (int): The number of CPU cores to use for training
        input_size (int): The input dimension fed into first Convolutional layer
    """

    def __init__(self, ngpu, input_size=768):
        super(TextClassifier, self).__init__()
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
                                  torch.nn.Linear(128 , 128))
    def forward(self, input):
        """Returns prediction on the features using the defined neural network"""
        x = self.main(input)
        return x

class CombinedModel(torch.nn.Module):
    """This combined model class starts of with the resnet50 pretrained model where only the fully connected last layer is unfreezed 
    for training the model weights. The rest are frozen which helps avoid overfitting since we do not have much data.
    The last layer will be a linear layer using the output from the resnet50 concatenated with the output from the text classifier model (256)
    as input and outputs the number of product categories. The torch Sequential module is used to connect the resnet50 and linear layer.
    
    Args:
        ngpu (int): The number of CPU cores to use for training
        num_classes (int): The number of classes to predict by the trained model
        input_size (int): The size of the input layer or Bert word embeddings fed to the text model
    """

    def __init__(self, ngpu, num_classes, input_size: int = 768):
        super(CombinedModel, self).__init__()
        self.ngpu = ngpu
        resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        for param in resnet50.parameters(): 
            param.requires_grad = False
        # unfreeze last layer params
        for param in resnet50.fc.parameters():
            param.requires_grad = True

        # for param in resnet50.layer4.parameters():
        #     param.requires_grad = True # This tends to result in overfitting and we need to use scheduler to decreasing learning rate per epoch
        # We will just not use this in the combined model unlike when using for image classification

        out_features = resnet50.fc.out_features
        self.image_classifier = torch.nn.Sequential(resnet50, torch.nn.Linear(out_features, 128))
        self.text_classifier = TextClassifier(ngpu=ngpu, input_size=input_size)
        final_layer = torch.nn.Linear(256, num_classes)
        self.main = torch.nn.Sequential(final_layer)

    def forward(self, image_features, text_features):
        """Returns prediction on the features using the defined neural network"""
        image_features = self.image_classifier(image_features)
        text_features = self.text_classifier(text_features)
        combined_features = torch.cat((image_features, text_features), dim=1)
        combined_features = self.main(combined_features)

        return combined_features

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
        for _, (images, text,  labels) in progress_bar:
            images, text, labels = images.to(device), text.to(device), labels.to(device)
            output = model(images, text)
            loss = loss_function(output, labels)
            accuracy = (torch.sum(torch.argmax(output, dim=1) == labels).item()) / len(labels)
            hist_val_acc.append(accuracy)
            hist_val_loss.append(loss.item())
            progress_bar.set_description(f"Validation metrics: acc = {round(float(accuracy), 4)}. mean_val_acc = {round(np.mean(hist_val_acc), 4)}. mean_val_loss = {round(np.mean(hist_val_loss), 4)}")
            loss_total += loss.item()

    return loss_total / len(valid_loader), round(np.mean(hist_val_acc), 5)
  

def train(model, device, train_loader, valid_loader, epochs=10, curr_epoch_num=0, prev_val_acc=0):

    """This function trains the CNN model, loops through the training data for a set number of epochs, 
    calculates predictions and loss, updates gradients and model parameters, prints loss in Tensorboard, saves the model every few epochs,
    and returns the trained CNN model.

    Args:
        model (Model): CNN model 
        device (device): Whether we are working on GPU or CPU
        train_loader (DataLoader): Training data
        valid_loader (DataLoader): Validation data
        epochs (int): Number of times we loop through training data to improve our model parameters
        curr_epoch_num (int): Number of epochs the model has already trained on 
        prev_val_acc (float): Maximum accuracy model has reached in previous epochs of training

    Returns:    
        model (Model): Trained CNN model
    """
     # Early stopping
    early_stopping = EarlyStopping(patience=4)
    last_loss = np.inf
    running_corrects = 0


    final_models_path = 'final_models'
    paths = [final_models_path]
    
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
            

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=3e-5)
    writer = SummaryWriter()
    batch_idx = 0
    total = 0 
    

    for epoch in range(curr_epoch_num, epochs+1):
        print('\n')
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        hist_acc = [] 
        model.train()
        for _, (images, text,  labels) in progress_bar:
            images, text, labels = images.to(device), text.to(device), labels.to(device)
            prediction = model(images, text)
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
            progress_bar.set_description(f"Epoch = {epoch}/{epochs}. acc = {round(float(accuracy), 4)}. mean_train_acc = {round(np.mean(hist_acc), 4)}. Loss = {round(float(loss), 4)}")

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
            torch.save({'epoch': epoch, 'val_acc': val_acc, 'model_state_dict': copy.deepcopy(model.state_dict())}, f'{final_models_path}/combined_model.pt')
                

        if early_stopping.early_stop:
            print("Early stopping invoked! We are at epoch:", epoch)
            return model

    print('End of epochs reached with best model saved as combined_model.pt')
    torch.save(model, 'final_combined_model.pt') # save the final model as well for comparison 
    return model
        

if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ngpu = 2
    dataset = ImageTextDataset()
    num_classes = dataset.num_classes

    if os.path.exists('image_text_train_data.pkl'):
        train_loader = pd.read_pickle('image_text_train_data.pkl')
        test_loader = pd.read_pickle('image_text_test_data.pkl')
        valid_loader = pd.read_pickle('image_text_validation_data.pkl')

    else:
        train_loader, valid_loader, test_loader = split_dataset(dataset, 0.8, 0.1) # 0.1 is validation of trainig data, 0.2 is testing data

    if os.path.exists('final_models/combined_model.pt'):
        further_training = input('Do you want the model to be trained further: ')
        model = CombinedModel(ngpu=ngpu, num_classes=num_classes)
        checkpoint = torch.load('final_models/combined_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        if further_training.lower() == 'yes': # If we want to train further in the future
            model_cnn = train(model, device, train_loader, valid_loader, epochs=55, curr_epoch_num=checkpoint['epoch']+1, prev_val_acc=checkpoint['val_acc'])

        model.eval()
        acc = test_accuracy(test_loader, device, model)
        print('Accuracy of the network on the test data (combined image and text data): {} %'.format(acc))

    else:
        model = CombinedModel(ngpu=ngpu, num_classes=num_classes)
        model.to(device)
        model_cnn = train(model, device, train_loader, valid_loader, epochs=50)
        acc = test_accuracy(test_loader, device, model_cnn)
        print('Accuracy of the network on the test data (combined image and text data): {} %'.format(acc))
