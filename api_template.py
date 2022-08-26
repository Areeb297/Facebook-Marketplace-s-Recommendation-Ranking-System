import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from pydantic import BaseModel
import pandas as pd
from torchvision.models import ResNet50_Weights
from torchvision import models
from combined_model import TextClassifier
##############################################################
# TODO                                                       #
# Import your image and text processors here     
from image_processor import ImageProcessor
from text_processor import TextProcessor
##############################################################



class Text_Classifier(nn.Module):
    def __init__(self,
                 decoder: dict = None,
                 num_classes: int = 2,
                 input_size: int = 768):
        super(Text_Classifier, self).__init__()
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

##############################################################
# TODO                                                       #
# Populate the __init__ method, so that it contains the same #
# structure as the model you used to train the text model    #
##############################################################
        
        self.decoder = decoder
    def forward(self, text):
        x = self.main(text)
        return x

    def predict(self, text):
        with torch.no_grad():
            x = self.forward(text)
            return x
    
    def predict_proba(self, text):
        with torch.no_grad():
            probabilities = torch.nn.functional.softmax(self.forward(text))
            return probabilities.flatten().tolist()


    def predict_classes(self, text):
        with torch.no_grad():
            predictions = self.forward(text)
            return self.decoder[int(torch.argmax(predictions, dim=1))]

    
class ImageClassifier(nn.Module):
    def __init__(self,
                num_classes: int = 2, 
                 decoder: dict = None):
        super(ImageClassifier, self).__init__()

##############################################################
# TODO                                                       #
# Populate the __init__ method, so that it contains the same #
        resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in resnet50.parameters(): 
            param.requires_grad = False
        # unfreeze last layer params
        for param in resnet50.fc.parameters():
            param.requires_grad = True

        for param in resnet50.layer4.parameters():
            param.requires_grad = True

        out_features = resnet50.fc.out_features
        self.linear = torch.nn.Linear(out_features, num_classes)
        self.main = torch.nn.Sequential(resnet50, self.linear)
# structure as the model you used to train the image model   #
##############################################################
        
        self.decoder = decoder

    def forward(self, image):
        x = self.main(image)
        return x

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

    def predict_proba(self, image):
        with torch.no_grad():
            probabilities = torch.nn.functional.softmax(self.forward(image))
            return probabilities.flatten().tolist()


    def predict_classes(self, image):
        with torch.no_grad():
            predictions = self.forward(image)
            return self.decoder[int(torch.argmax(predictions, dim=1))]


class CombinedModel(nn.Module):
    def __init__(self,
                 num_classes: int = 2, 
                 decoder: list = None):
        super(CombinedModel, self).__init__()
##############################################################
# TODO                                                       #
        resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in resnet50.parameters(): 
            param.requires_grad = False
        # unfreeze last layer params
        for param in resnet50.fc.parameters():
            param.requires_grad = True

        out_features = resnet50.fc.out_features
        self.image_classifier = torch.nn.Sequential(resnet50, torch.nn.Linear(out_features, 128))
        self.text_classifier = TextClassifier()

        final_layer = torch.nn.Linear(256, num_classes)
        self.main = torch.nn.Sequential(final_layer)
# Populate the __init__ method, so that it contains the same #
# structure as the model you used to train the combined model#
##############################################################
        
        self.decoder = decoder

    def forward(self, image_features, text_features):
        image_features = self.image_classifier(image_features)
        text_features = self.text_classifier(text_features)
        combined_features = torch.cat((image_features, text_features), dim=1)
        combined_features = self.main(combined_features)
        return combined_features

    def predict(self, image_features, text_features):
        with torch.no_grad():
            combined_features = self.forward(image_features, text_features)
            return combined_features
    
    def predict_proba(self, image_features, text_features):
        with torch.no_grad():
            probabilities = torch.nn.functional.softmax(self.forward(image_features, text_features))
            return probabilities.flatten().tolist()

    def predict_classes(self, image_features, text_features):
        with torch.no_grad():
            predictions = self.forward(image_features, text_features)
            return self.decoder[int(torch.argmax(predictions, dim=1))]



# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str


try:
##############################################################
# TODO     
    decoder = pd.read_pickle('image_decoder.pkl')  
    n_classes = len(decoder)
    text_model = Text_Classifier(decoder, num_classes=n_classes)
    checkpoint = torch.load('text_model.pt', map_location='cpu')
    text_model.load_state_dict(checkpoint['model_state_dict'])    
    device = torch.device('cpu')
    text_model.to(device)
    text_model.eval()                                          #
# Load the text model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the text model    #
# you used for training it, and then load the weights in it. #
# Also, load the decoder dictionary that you saved as        #
# text_decoder.pkl                                           #
##############################################################
    pass
except:
    raise OSError("No Text model found. Check that you have the decoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
    decoder = pd.read_pickle('image_decoder.pkl')  
    n_classes = len(decoder)
    image_model = ImageClassifier(num_classes=n_classes, decoder=decoder)
    checkpoint = torch.load('final_models/image_model.pt', map_location='cpu')
    image_model.load_state_dict(checkpoint['model_state_dict'])    
    device = torch.device('cpu')
    image_model.to(device)
    image_model.eval()    
# Load the image model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the image model   #
# you used for training it, and then load the weights in it. #
# Also, load the decoder dictionary that you saved as        #
# image_decoder.pkl                                          #
##############################################################
    pass
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
    decoder = pd.read_pickle('image_decoder.pkl')  
    n_classes = len(decoder)
    combined_model = CombinedModel(num_classes=n_classes, decoder=decoder)
    checkpoint = torch.load('final_models/combined_model.pt', map_location='cpu')
    combined_model.load_state_dict(checkpoint['model_state_dict'])    
    device = torch.device('cpu')
    combined_model.to(device)
    combined_model.eval()    
# Load the combined model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the combined model#
# you used for training it, and then load the weights in it. #
# Also, load the decoder dictionary that you saved as        #
# combined_decoder.pkl                                       #
##############################################################
    pass
except:
    raise OSError("No Combined model found. Check that you have the encoder and the model in the correct location")

try:
##############################################################
# TODO       
    text_processor = TextProcessor()                                                 #
# Initialize the text processor that you will use to process #
# the text that you users will send to your API.             #
# Make sure that the max_length you use is the same you used #
# when you trained the model. If you used two different      #
# lengths for the Text and the Combined model, initialize two#
# text processors, one for each model                        #
##############################################################
    pass
except:
    raise OSError("No Text processor found. Check that you have the encoder and the model in the correct location")

try:
##############################################################
# TODO                                                       #
    image_processor = ImageProcessor()
# Initialize the image processor that you will use to process#
# the text that you users will send to your API              #
##############################################################
    pass
except:
    raise OSError("No Image processor found. Check that you have the encoder and the model in the correct location")

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

@app.post('/predict/text')
def predict_text(text: str = Form(...)):
  
    ##############################################################
    # TODO 
    processed_text = text_processor(text)
    # Process the input and use it as input for the text model   #
    # text.text is the text that the user sent to your API       #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################

    return JSONResponse(content={
        "Category": text_model.predict_classes(processed_text), # Return the category here
        "Probabilities": text_model.predict_proba(processed_text) # Return a list or dict of probabilities here
            })
  
  
@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)
    processed_image = image_processor(pil_image)

    ##############################################################
    # TODO                                                       #
    # Process the input and use it as input for the image model  #
    # image.file is the image that the user sent to your API     #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################

    return JSONResponse(content={
    "Category": image_model.predict_classes(processed_image), # Return the category here
    "Probabilities": image_model.predict_proba(processed_image) # Return a list or dict of probabilities here
        })
  
@app.post('/predict/combined')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    print(text)
    pil_image = Image.open(image.file)
    
    ##############################################################
    # TODO                                                       #
    pil_image = Image.open(image.file)
    processed_image = image_processor(pil_image)
    processed_text = text_processor(text)
    # Process the input and use it as input for the image model  #
    # image.file is the image that the user sent to your API     #
    # In this case, text is the text that the user sent to your  #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################

    return JSONResponse(content={
    "Category": combined_model.predict_classes(processed_image, processed_text), # Return the category here
    "Probabilities": combined_model.predict_proba(processed_image, processed_text) # Return a list or dict of probabilities here
        })
    
    
if __name__ == '__main__':
  uvicorn.run(app, host="127.0.0.1", port=8080)