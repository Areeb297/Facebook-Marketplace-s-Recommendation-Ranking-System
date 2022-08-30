from clean_tabular_data import clean_text_data
import torch
import pandas as pd
from transformers import BertTokenizer
from transformers import BertModel

class TextProcessor():
    '''
    The TextProcessor class takes in a sentence relating to a product, applies the cleaning function we used before to
    clean the description column (remove non-alphanumeric characters etc), tokenizes the text, uses Bert to create word embeddings
    where the max_length is the maximum tokens accepted in a sequence (of every description), and returns encoded text with shape 
    shape (1, 768, 50) 1 is the batch size, 768 is the embedding vector size of each token and 50 is the number of tokens per description.


    Args:
        max_length (int): Maximum length of number of tokens generated in every description

    '''
    def __init__(self, max_length=20):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model.eval()
        self.max_length = max_length

    def __call__(self, description): 
        '''
        This method runs only when the class instance is called 

        Args:
            description (str): Sample product description text relating to a product category
        '''
        description = clean_text_data(pd.Series(description))[0]
        encoded = self.tokenizer.batch_encode_plus([description], max_length=self.max_length, padding='max_length', truncation=True) # every description in list format
        encoded = {key: torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)
        description = description.reshape(1, -1, self.max_length)
        return description
                
# class Classifier(torch.nn.Module):
#     """This CNN class uses the torch Sequential module to build layers of the text data neural network which includes 
#     convolutional layers, dropout layers, max pooling layers, linear layers with ReLU as activation functions, with
#     the last layer having 13 outputs for each category class in the dataset.
    
#     Args:
#         ngpu (int): The number of CPU cores to use for training
#         num_classes (int): The number of classes to predict by the trained model
#         input_size (int): The input dimension fed into first Convolutional layer
    
#     """

#     def __init__(self, ngpu, num_classes=13, input_size=768):
#         super(Classifier, self).__init__()
#         self.ngpu = ngpu
#         self.main = torch.nn.Sequential(torch.nn.Conv1d(input_size, 256, kernel_size=3, stride=1, padding=1),
#                                   torch.nn.ReLU(),
#                                   torch.nn.MaxPool1d(kernel_size=2, stride=2),
#                                   torch.nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
#                                   torch.nn.ReLU(),
#                                   torch.nn.MaxPool1d(kernel_size=2, stride=2),
#                                   torch.nn.Dropout(p=0.2),
#                                   torch.nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
#                                   torch.nn.Dropout(p=0.2),
#                                   torch.nn.ReLU(),
#                                   torch.nn.MaxPool1d(kernel_size=2, stride=2),
#                                   torch.nn.Dropout(p=0.2),
#                                   torch.nn.ReLU(),
#                                   torch.nn.Flatten(),
#                                   torch.nn.Linear(128 , 64),
#                                   torch.nn.ReLU(),
#                                   torch.nn.Linear(64, num_classes))
#     def forward(self, input):
#         """Returns prediction on the features using the defined neural network"""
#         x = self.main(input)
#         return x

# if __name__ == '__main__':
#     sample_text = 'This is a new sentence relating to product category bicycle with 2 wheels only'
#     process = TextProcessor()
#     description = process(sample_text)
#     model = Classifier(ngpu=2, num_classes=13)
#     checkpoint = torch.load('text_model.pt')
#     model.load_state_dict(checkpoint['model_state_dict'])
#     device = torch.device('cpu')
#     model.to(device)
#     model.eval()
#     decoder = pd.read_pickle('decoder.pkl')
#     output = model(description)
#     _, predicted = torch.max(output.data, 1)
#     pred = decoder[int(predicted)]
#     print(pred)


    
