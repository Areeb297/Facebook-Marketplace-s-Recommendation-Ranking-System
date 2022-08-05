from merge_data import merge_im_array
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch
import pickle
import numpy as np
from image_classification import category_encode_decode

def generate_features_targets():
    """This function checks whether the pickle file for the CNN dataframe exists and if not, creates the dataframe,
    retains only the main category of the product, flattens the image array column, encodes the category into numbers and,
    saves the encoding, features (image array column), and targets (encoded category column) as pickle files.
    """

    if not os.path.exists('CNN_merged_dataframe.pkl'):
        merge_im_array('cleaned_images/', 'CNN_merged_dataframe.pkl')


    if not os.path.exists('CNN_features.pkl'): # cehcks if the features pickle exists to see if we already ran this function
        df = pd.read_pickle('CNN_merged_dataframe.pkl')
        df.category = df.category.apply(lambda x: x.split('/')[0]) # Retain only the first category

        decode, df = category_encode_decode(df)
        with open('CNN_decoder.pkl', 'wb') as file:
            pickle.dump(decode, file)

        y = df.category_codes # target variable
        with open('CNN_targets.pkl', 'wb') as file:
            pickle.dump(y, file)

        X = df['image_array'].apply(lambda x: x.flatten())

        with open('CNN_features.pkl', 'wb') as file:
            pickle.dump(X, file)


class ProductDataset(Dataset):

    """This class inherits from the Dataset module, obtains all the features and targets from the pickle files, 
    checks they are the same lengths and lastly has magic methods for indexing and returning the length of the dataset."""

    def __init__(self):
        super().__init__()
        self.X = pd.read_pickle('CNN_features.pkl')
        self.y = pd.read_pickle('CNN_targets.pkl')
        assert len(self.X) == len(self.y)
    
    def __getitem__(self, index):
        """Returns the features and labels of a product given an index where the features are given in shape [channels, height, width].
        Args:
            index (int): The row index of the desired product in the data
        Returns:
            tuple: Tuple containing features and labels for a product
        """
        features = self.X.iloc[index] # I need the row index (both columns and rows are numbers)
        labels = torch.tensor(self.y.iloc[index]).long()
        features = torch.tensor(features).float()
        features = features.reshape(3, 155, 155)
        transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # 3 color channels to normalize
        features = transform(features)
        # We can add random rotation or horizontal flip to our feature images e.g angle 0-180
        # features = rotate(features, random.randrange(0, 180))
        # flip = transforms.RandomHorizontalFlip(p=1) # p = probability
        # features = flip(features)
        return (features, labels)

    def __len__(self):
        """Returns the length of the dataset containing predictors."""
        return len(self.X)


class CNN(torch.nn.Module):
    """This CNN class uses the torch Sequential module to build layers of the neural network which includes 
    convolutional layers, dropout layers, max pooling layers, a linear layer with ReLU as activation functions, and
    softmax being used at the end to output probabilities of each class in the dataset."""

    def __init__(self):
        super().__init__()
        # Declaring the Architecture
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 200, 5, 2), # Kernel size 7 with stride 2
            torch.nn.MaxPool2d(4,4),
            torch.nn.Dropout(p=0.1),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(200, 100, 3),
            # torch.nn.MaxPool2d(2, 2), # Max pooling (2, 2) filter
            torch.nn.Dropout(p=0.1),
            torch.nn.ReLU(),

            torch.nn.Conv2d(100, 50, 3),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout(p=0.3),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(2450, 100),
            torch.nn.Linear(100, 13), #  Predicting 13 product categories
            torch.nn.Softmax(dim=1)
        )

    def forward(self, features):  
        """Returns prediction on the features using the defined neural network""" 
        return self.layers(features)


def split_dataset(dataset):

    """This function splits the dataset into train, validation, and test and uses dataloader to store the values as iterables
    for all three datasets.

    Args:
        dataset (Dataset): The dataset of features and labels as tuples and torch tensor format
    
    Returns:
        Tuple: Tuple containing train, validation, and test data in form of dataloader objects.
    """

    train_size = 9000
    test_size = int(len(dataset) - train_size)
    valid_size = int(train_size * 0.1)

    train_data, test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(100))
    train, validation = random_split(train_data, [(train_size - valid_size), valid_size], generator=torch.Generator().manual_seed(100))

    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    valid_loader = DataLoader(validation, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    
    for file_name, dataloader in [('train_data.pkl', train_loader), ('test_data.pkl', test_loader), ('validation_data.pkl', valid_loader)]:
        with open(file_name, 'wb') as file:
            pickle.dump(dataloader, file)

    return train_loader, valid_loader, test_loader


# validation 
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
    
    model.eval() # it tells your model that you are testing the model
    loss_total = 0

    # Test validation data
    with torch.no_grad(): # Deactivate autograd, requires_grad is set to False, do not calculate gradients of new variables as we testing only
        for data in valid_loader:
            features, labels = data
            features, labels = features.to(device), labels.to(device)

            output = model(features)
            loss = loss_function(output, labels)
            loss_total += loss.item()

    return loss_total / len(valid_loader)

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
        if curr_val_loss > previous_val_loss:
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
    
    accuracy = torch.div(100 * correct.double(), total, rounding_mode='floor')

    print('Accuracy of the network on the test images: {} %'.format(accuracy))
    return accuracy



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
    early_stopping = EarlyStopping(patience=6)
    last_loss = np.inf
    # patience = 2
    # triggertimes = 0
    
    optimiser = torch.optim.SGD(model.parameters(), lr=0.009)
    # weight decay adds penalty to loss function, shrinks weights during backpropagation to prevent overfitting and exploding gradients
    writer = SummaryWriter()
    batch_idx = 0

    for epoch in tqdm(range(epochs+1)):
        loss_per_epoch = 0
        model.train()
        for i, batch in enumerate(train_loader):
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            prediction = model(features)
            loss = F.cross_entropy(prediction, labels)
            loss.backward()

            if i % 10 == 0:
                print(f'Loss: {loss.item()}') # print loss every 10th value in batch

            optimiser.step()
            optimiser.zero_grad()
            # exp_lr_scheduler.step()
            writer.add_scalar('Training Loss', loss.item(), batch_idx)
            loss_per_epoch += loss.item()
            batch_idx += 1
        
        print(f'epoch number {epoch} with average loss: {loss_per_epoch / len(train_loader)}')


        # Early stopping
        validation_loss = validation(model, device, valid_loader, F.cross_entropy) 
        early_stopping(last_loss, validation_loss)
        writer.add_scalar('Validation Loss', validation_loss, batch_idx)
        last_loss = validation_loss


        if early_stopping.early_stop:
            print("We are at epoch:", epoch)
            return model
        
        if epoch % 5 == 0:
            torch.save(model, 'CNN_model.pth')
        # torch.save(model.state_dict(), 'model_cnn')
    
    return model

        # if validation_loss > last_loss:
        #     trigger_times += 1
        #     print('Trigger Times:', trigger_times)

        #     if trigger_times >= patience:
        #         print('Early stopping!\nStart to test process.')
        #         return model

        # else:
        #     print('trigger times: 0')
        #     trigger_times = 0

        # last_loss = validation_loss




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generate_features_targets()

    if os.path.exists('train_data.pkl'):
        train_loader = pd.read_pickle('train_data.pkl')
        test_loader = pd.read_pickle('test_data.pkl')
        valid_loader = pd.read_pickle('validation_data.pkl')

    else:
        dataset = ProductDataset()
        train_loader, valid_loader, test_loader = split_dataset(dataset)

    if os.path.exists('CNN_model.pth'):
        model_cnn = torch.load('CNN_model.pth')
        model_cnn.eval()
        acc = test_accuracy(test_loader, device, model_cnn)

    else:

        model = CNN()
        model.to(device)
        model_cnn = train(model, device, train_loader, valid_loader, epochs=20)
        acc = test_accuracy(test_loader, device, model_cnn)
