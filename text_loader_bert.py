# Text data loader

import enum
import torchvision.transforms as transforms
import os
import torch
import pandas as pd
from transformers import BertTokenizer
from transformers import BertModel


class Textloader(torch.utils.data.Dataset):
    '''
    The Textloader object inherits its methods from the
    torch.utils.data.Dataset module.
    It reads and loads the product description and category codes from the merged data from before, 
    dropping duplicated rows as image id column is not needed. We use BertTokenizer to transform descriptions to 
    use as input in the BertModel. We encode every description into embeddings taking a fixed max length of tokens
    and when indexing through this class, it returns the description and corresponding label in tuple form. 

    Args:
        root_dir (str): Directory where the merged and cleaned data of products and images is stored
        max_length (int): Maximum length of number of tokens generated in every description

    '''

    def __init__(self, root_dir: str = 'product_images.csv', max_length: int = 20):
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f'The file {self.root_dir} does not exist')

        self.merged_data = pd.read_csv(root_dir)
        self.merged_data.drop_duplicates(subset=['product_description', 'category_codes'], inplace=True) #  not using image id anymore so we can drop duplicate rows
        self.description = self.merged_data['product_description'].to_list()
        self.labels = self.merged_data['category_codes'].to_list()

        self.num_classes = len(set(self.labels))
        self.decoder = pd.read_pickle('image_decoder.pkl') # read in the decoder file which we saved from image classification

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.max_length = max_length 

    def __getitem__(self, index):
        label = self.labels[index]
        label = torch.as_tensor(label).long()   
        description = self.description[index]
        encoded = self.tokenizer.batch_encode_plus([description], max_length=self.max_length, padding='max_length', truncation=True) # every description in list format
        encoded = {key: torch.LongTensor(value) for key, value in encoded.items()}

        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)

        description = description.squeeze(0) # remove unnecessary dimension that the CNN model will not use
        return description, label

    def __len__(self):
        return len(self.labels)


# if __name__ == '__main__':
#     dataset = Textloader()
#     print(dataset.num_classes)
#     print(dataset[10], dataset.decoder[int(dataset[10][1])])
#     print(dataset[20][0].size())
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=True)
    # for i, (data, labels) in enumerate(dataloader):
    #     print(data)
    #     print(labels)
    #     print(data.size())
    #     break# Text data loader

import enum
import torchvision.transforms as transforms
import os
import torch
import pandas as pd
from transformers import BertTokenizer
from transformers import BertModel


class Textloader(torch.utils.data.Dataset):
    '''
    The Textloader object inherits its methods from the
    torch.utils.data.Dataset module.
    It reads and loads the product description and category codes from the merged data from before, 
    dropping duplicated rows as image id column is not needed. We use BertTokenizer to transform descriptions to 
    use as input in the BertModel. We encode every description into embeddings taking a fixed max length of tokens
    and when indexing through this class, it returns the description and corresponding label in tuple form. 

    Args:
        root_dir (str): Directory where the merged and cleaned data of products and images is stored
        max_length (int): Maximum length of number of tokens generated in every description

    '''

    def __init__(self, root_dir: str = 'product_images.csv', max_length: int = 20):
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f'The file {self.root_dir} does not exist')

        self.merged_data = pd.read_csv(root_dir)
        self.merged_data.drop_duplicates(subset=['product_description', 'category_codes'], inplace=True) #  not using image id anymore so we can drop duplicate rows
        self.description = self.merged_data['product_description'].to_list()
        self.labels = self.merged_data['category_codes'].to_list()

        self.num_classes = len(set(self.labels))
        self.decoder = pd.read_pickle('image_decoder.pkl') # read in the decoder file which we saved from image classification

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.max_length = max_length 

    def __getitem__(self, index):
        label = self.labels[index]
        label = torch.as_tensor(label).long()   
        description = self.description[index]
        encoded = self.tokenizer.batch_encode_plus([description], max_length=self.max_length, padding='max_length', truncation=True) # every description in list format
        encoded = {key: torch.LongTensor(value) for key, value in encoded.items()}

        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)

        description = description.squeeze(0) # remove unnecessary dimension that the CNN model will not use
        return description, label

    def __len__(self):
        return len(self.labels)


# if __name__ == '__main__':
#     dataset = Textloader()
#     print(dataset.num_classes)
#     print(dataset[10], dataset.decoder[int(dataset[10][1])])
#     print(dataset[20][0].size())
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=True)
    # for i, (data, labels) in enumerate(dataloader):
    #     print(data)
    #     print(labels)
    #     print(data.size())
    #     break
