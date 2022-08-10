# FB Marketplace Recommendation Ranking System

## Milestone 1: An overview of the system

> This project works on building and closely replicating a product ranking system of Facebook Marketplace to provide users with the most relevant products based on their search query. This is done via a multimodal pre-trained deep neural network using Transfer Learning (Text model + Image model). The model is a mini-implementation of a larger system that Facebook developed to generate product recommendation rankings for buyers on Facebook Marketplace. Shown below is a flowchart describing the overview of the system encompassing various technologies:

![image](https://user-images.githubusercontent.com/51030860/178149528-8a7c5b0c-3f14-46b0-b708-ff3faf455755.png)

Here is the ![video link](https://www.youtube.com/watch?v=1Z5V2VrHTTA&ab_channel=AiCore) for further information and reference.

## Milestone 2: Cleaning the tabular and image datasets

- In this stage, we perform data cleaning steps for the product and image datasets. Firstly, concerning the product tabular dataset (Products.csv), we have built a pipeline which completes all cleaning steps like ensuring all null and duplicate values are removed and all data formats are correct e.g., the data type of the price column is converted to float and the time column is converted to datetime. We ensure features like location and product category are converted into 'category' format (nominal data). Additionally, we clean the text data by removing non-alphanumeric characters and unnecessary whitespaces which will help in keeping only important words when using TF-IDF and linear regression to predict product price. For these transformations, we use the pandas library and Regex expression to clean the product dataset as shown below.

The code for this cleaning pipeline can be found in "clean_tabular_data.py"
```python

non_alpha_numeric = column.str.replace('\W', ' ', regex=True).apply(lambda x: x.lower()) # remove non-alphanumeric characters
non_whitespace = non_alpha_numeric.str.replace('\s+', ' ', regex=True) # For removing unnecessary whitespace
price_column.str.replace('[^0-9.]', '', regex=True).astype('float64') # Keep only numbers and decimals in the price column using regex expression

# remove all single characters
clean_text = non_whitespace.apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))
# For product name, we want to shorten the longest names and therefore, we use the code below to number of words to a minimum
if keep_char != None:
    return clean_text.apply(lambda x: ' '.join(x.split(' ')[0:keep_char])) # keep a certain number of words
return clean_text

# We get duplicates with only these columns as our criteria (any product with same values for these columns has to be a duplicate)
and keep only the first occuring values

data.drop_duplicates(subset=['product_name', 'location', 'product_description', 'create_time', 'price'], keep='first', inplace=True)

# remove unnecessary columns
data.drop(columns=['url', 'page_id'], inplace=True)
```

- Concerning the images folder and Images.csv file, we create a cleaning pipeline for both. For the images.csv folder, we will discuss more in the next milestone.

- Regarding the images folder, we develop functions to resize all the images into one consistent format with same number of channels (3) (RGB) and size (height, width). As using high pixel size will result in memory problems and poor performance when running sklearn ML models, we take two approaches when cleaning the images folder. For machine learning, we use a black background image of size (30x30) and paste the images on that after scaling to the appropriate dimensions. We save these (3, 30, 30) images in the 'cleaned_images_ML' folder with the name of each jpg file being the image id. 

- For generating images for our CNN model, we obtain images of size (3, 154, 154) as this was based on finding minimum height and width from the image dataset, using the lowest number from them and subtracting one if the number was odd. We do not run into memory problems as CNNs are designed to be working with large image data. We save the images in the 'cleaned_images' folder. 

We can find the code in the file 'clean_images.py' where code snippets are shown below as to how the images were resized and transformed:

```python

# For the scikit learn classification dataset, we use pixel size 30x30:

final_size = 30
size = im.size # size of image
ratio = float(final_size) / max(size)  # We calculate the ratio to change aspect ratio or condense image
new_image_size = tuple([int(x*ratio) for x in size]) 

im = im.resize(new_image_size, Image.Resampling.LANCZOS) # We resize to the new image size (ratio * previous_size)
new_im = Image.new("RGB", (final_size, final_size)) # new_img_size = (int(ratio * prev_size[0]), int(ratio * prev_size[1])) # RGB black background
new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2)) # paste on top of black background such that image is in center

list_img = glob.glob('images/*.jpg') # Get all files with file type .jpg (glob library gets filepath to image)

# For the CNN dataset, we calculate minimum height and width of all images (and subtract 1 if odd):

min_width = 100000
min_height = 100000

for img in list_img:
    size = Image.open(img).size
    width, height = size[0], size[1]

    if width < min_width:
        min_width = width

    if height < min_height:
        min_height = height

   min_dim = min(min_height, min_width)
    if min_dim % 2 != 0:
        min_dim -=1 # We want even pixel size

# check rgb and if not (4 dimensions etc or grayscale), convert mode to RGB

    new_img = Image.open(img)
    # check RGB color/mode
    if new_img.mode != 'RGB':
        new_img = new_img.convert('RGB')
```

- One important point to mention is that every product can have more than one corresponding image, hence we need to merge the image and product datasets using image id as the join column when predicting product categories using image data (multiclass classification). We will see more on this in the next milestone.


## Milestone 3: Create simple Machine Learning models

We created two simple ML models using Linear regression (Regression) and Logistic regression (for multi-class classifiation):

- Predicting the product price based on product name, description, and location (Linear Regression)
- Predicting product category based on the image dataset converted into numpy arrays (Logistic Regression)

<ins>1 - Linear Regression model for predicting price (Regression):</ins>

- First we split the data into features (name, description, location) and targets (price) to then transforming our features using TfidfVectorizer. We convert all the text into weights assigned to each word based on their term frequency in the whole product dataframe after cleaning it ('clean_tabular_data.py'). Additionally, we exclude stopwords from our features such as 'the', 'are' etc using stopwords from the nltk.corpus library. This process is done to remove unnecessary words from hindering our model performance. 

- Moreover, we have hyperparameters we define for Gridsearch to select the optimal of them such as n_gram range of tfidf vector and minimum term frequency to include a word. Lastly we perform linear regression where we do get a terrible RMSE (approx 86,000) and r^2 score (-60) as we have too many features (curse of dimensionality) and have overparametrized our model. We can potentially focus on removing further words from our model resulting in removal of more feature columns, or obtain more data in the future. We can try other models like random forest regressor but they take a long time to fit with so many features and hence may not be feasible at the moment. Furthermore, we may only keep the first few words in the product name to avoid having a seriously long product names in our analysis as mentioned before. Shown below are code snippets from the 'regression.py' file about how we transform text into numerical data using Tf-idf and use grid search.

```python
# After cleaning Product.csv:
X = product_data[['product_name', 'product_description', 'location']] # features
y = product_data['price'] # targets

stop = set(stopwords.words('english')) # Get the stopwords list
# Initialize tfidf vectors to transform all text columns
tfidf = ColumnTransformer([
        ("vector_1", TfidfVectorizer(stop_words=stop), 'product_name'),
        ("vector_2", TfidfVectorizer(stop_words=stop), 'product_description'),
        ("vector_3", TfidfVectorizer(stop_words=stop), 'location')], 
        remainder='passthrough') 

# Create a pipeline to run the column transformations and perform linear regression
pipeline = Pipeline([   
        ("tfidf", tfidf),
        ("lr", LinearRegression())])

# hyperparameters for the tfidf vectors to tune and optimise
parameters = {
    'tfidf__vector_1__ngram_range': ((1, 1), (1, 2)),
    'tfidf__vector_2__min_df': (0.005,  0.2, 0.01)}

# Find the best hyperparameters for both the feature extraction and regressor using Grid Search from sklearn
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2)

# split data in to train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
grid_search.fit(X_train, y_train)

# calculate and print RMSE (in regression.py, we also print R-squared representing the goodness of fit)
rmse = np.sqrt(mean_squared_error(y_test, grid_search.predict(X_test)))
print(f'RMSE: {rmse}')
```
<ins>2 - Logistic Regression for predicting product category (Classification):</ins>

- Coming back to merging the data first, we use pandas merge to combine both the Image.csv and Product.csv files after having cleaned them. For the images.csv file, we drop unecessary columns like create_time, bucket_link, image_ref and we merge both datasets on the common image id column. Once we have a merged dataframe, we firstly split the category data such that we only retain the one relevant category for every product observation to then convert the category into codes using cat.codes and then save the mapping as a dictionary into a pickle file ('image_decoder.pkl') to be able to decode them later once our model has returned the predictions. We save the cleaned product_images as product_images.csv file. 

- The final shape for this data is 11,115 rows with 8 columns which is not a lot of data when using machine learning or deep neural nets. With transfer learning however, we can achieve a high accuracy when implementing deep learning but we will examine that later. For our deep learning network, we will define an image loader class which will read in the product_images.csv file, using the image id from the dataset, look through the cleaned_images (154x154) folder for the required image id, convert to pytorch tensor, obtain the features, and so on. However, for machine learning, we will require another method to combine the images and the csv file as we do not have an image loader class.

- To do this, in our merge_data.py file, we create another function called 'generate_ml_classification_data' which initializes an empty column 'image_array' in the cleaned dataframe, loops through the images from the cleaned_images_ML (size 30x30) folder, converts them into numpy array format using np.asarray(image), takes the image id from the image file name (indexing and removing .jpg from the file name), checks which row of our merged dataframe corresponds to that image id, then places each image_array as a list in the correct row under the column 'image_array'. We save this dataframe as a pickle file (ML_product_images.pkl) to prevent the array format of images being changed to string format after reloading the dataframe.

- We can find the code for merging the data and the steps mentioned above in the 'merge_data.py' file. Some code snippets are shown below from that file:

```python
# merge the product and image data
df = pd.merge(left=products_data, right=images_data, on=('product_id', 'create_time'), sort=True) # merge on common columns (product_id, create_time)
df.drop(columns=['product_id','create_time', 'bucket_link', 'image_ref'], inplace=True) # Drop irrelevant columns

# Retain first category from group of relevant categories for each product e.g., 'Home & Garden' from 'Home & Garden / Other Goods / DIY Tools & Materials'
df.category = df.category.apply(lambda x: x.split('/')[0]) # Retain only the first category
df.category =  df.category.astype('category') # change datatype to category to be able to use cat.codes
df['category_codes'] =  df.category.cat.codes # convert category column into numerical codes
decoder_dict = dict(enumerate(df['category'].cat.categories)) # dictionary that contains encodings of text to numbers

df.to_csv('product_images.csv') # This will be saved for later to use for deep learning/CNNs/transfer learning

# convert the image data into array
data['image_array'] = " " # initalize empty column in dataframe
dirs = os.listdir('cleaned_images_ML')
for image_name in dirs:
 if image_name[:-4] in data['image_id'].values: # exclude the .jpg from the image name to get image id
 image = Image.open(path + image_name)
 arr_im = np.asarray(image) # convert to array
 data['image_array'].loc[data['image_id'] == item[:-4]] = [arr_im] # locate which column the image id matches, place the image array in that column
 
# We also use seaborn to plot a countplot of all the category classes which shows us the classes are fairly balanced and we do not need any oversampling etc
sns.countplot(y=data.category)
plt.show()
```
- We see that our image dataset is mostly balanced which indicates no issues regarding imbalanced set of classes when it comes to multiclassification.  We see as shown below that Home & Garden category has the most images/observations so it would make sense that the model performs the best on that category.

<p align="center">
  <img src="https://user-images.githubusercontent.com/51030860/182227206-cddce56e-0503-47f0-9bd6-007cd2e7d91d.png" alt="Sublime's custom image"/>
</p>

- Finally, moving onto our image classification model, we read in the pickle file we saved earlier ('ML_product_images.pkl'), take the image array column as our features by flattening each array so that we have a column of features (30x30 = 900 features), take the category codes column as our target column, perform train test split, define a param grid where we list hyperparameters of logistic regression to tune, fit and predict the logistic regression model on the data. Instead of Grid Search, we exploit randomized search to save time and optimize hyperparameters of the logistic regression function such as the max iterations and regularization (C hyperparameter) etc. We use the lbfgs solver as it is suitable for multiclassification. We did not use other suitable solvers like newton-cg, sag, saga as they take a lot of time to run. 

- Shown below are code snippets from the file 'ml_classification.py' indicating how the train test was implemented to split the data, randomized grid search used and predictions made by the logistic regression model.

```python
# Get features and targets from the dataframe
X = df['image_array'].apply(lambda x: x.flatten()) # Flatten array so every row contains one flattened array
y = df.category_codes

# Split the data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(list(X), y, test_size=0.3, random_state=42)

# Hyperparameters
param_grid = [{'penalty':['l2'], 'C': np.logspace(-4, 4, 30), 'solver': ['lbfgs'], 'max_iter': [400]}]

model = LogisticRegression() 

random_search = sklearn.model_selection.RandomizedSearchCV( # randomized search of hyperparameters in param_grid
    estimator=model, 
    param_distributions=param_grid,
    verbose=4, n_iter=2, cv=2)
        
# print predictions and classification report
print(f'The accuracy of our predictions: {round(accuracy_score(y_test, y_pred), 5) * 100} %')
print(classification_report(y_test, y_pred))
```
 
- The results we obtain are an accuracy of around 15% which is poor but it is better than say random guessing (1/13 ~ 7% accuracy) and gives us a benchmark to compare and improve upon when comparing using deep learning frameworks. The sklearn models are not designed to classify images so we expected poor performance. 

-  We print the classification report additionally which gives us the precision, recall, and f1-score for each category where as expected, we can see that our model performs more confidently when predicting the Home & Garden category, 'Computers & Software' and 'Office Furniture & Equipment' as these features have more data.
 
<p align="center">
<img src="https://user-images.githubusercontent.com/51030860/183982155-eac137fe-aba4-4067-9711-2cbdd6875bba.png" alt="Sublime's custom image"/>
</p>

- The results are as follows:
> Best parameters: {'solver': 'lbfgs', 'penalty': 'l2', 'max_iter': 400, 'C': 2.592943797404667}, the accuracy of our predictions: 15.20%

## Milestone 4: Creating a pytorch vision CNN model (without and with transfer learning)

<ins>2 - CNN without transfer learning:</ins>


- After testing with logistic regression, we use a CNN deep neural network instead to inspect whether our performance improves on the images we saved in higher pixel dimension (3x154x154). Coming back to the image loader class we referred to earlier, this class uses the product_images.csv file to obtain the category codes (labels) and the image id which then is used to locate the corresponding image file with the same id in the cleaned_images folder. The image is then resized to 128x128 and centered to run the CNN model faster (even though we will use GPU) applied random horizontal flips on the data to generate more variety in the images for better model performance, converted the images to pytorch tensors, and normalized the three channels in the image (RGB). Given an index, our class will then return the desired image tensor with its label in tuple form (image_tensor, label) and we can apply the decoder dictionary if we want to see what class the image tensor belongs to. We can find the code in the 'image_loader.py' file where shown below are code snippets about the class:

```python
self.merged_data = pd.read_pickle('product_images.csv') # Get the products_images data we saved before from the merge.py file
self.files = self.merged_data['image_id'] # get image id from the data
self.labels = self.merged_data['category_codes'] # Finally get the labels/category codes
self.decoder = pd.read_pickle('image_decoder.pkl') # read in the decoder dictionary save as a pickle file

self.transform = transforms.Compose([ # the transformation pipeline an image goes through
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3 channels, 3 means
                     std=[0.229, 0.224, 0.225]) # 3 channels, 3 standard deviations
])

label = self.labels[index] # get the label given an index from the whole data
label = torch.as_tensor(label).long()
image = Image.open('cleaned_images/' + self.files[index] + '.jpg') # get the corresponding image to the label
image = self.transform(image).float() # we will use this class to load data in batches using the Dataloader module from pytorch

# If we want to decode our image label, we use:
dataset = ImageDataset() # initialize class
dataset.decoder[int(dataset[0][1])] # the 0 corresponds to the image index, 1 corresponds to getting the label as the class returns a tuple (tensor, label)
```
- We define the dataloader function below which splits the data into train, test, and validation and returns them as dataloaders that will be looped through in batches of size 32. We first specify the training percentage, get the data, then subtract that from the whole data to get the test data. Finally, we define a validation percentage which we apply to the training data and split the data into training and validation. We use 10% of the training data as validation and 20% of the whole data as testing (0.8 for training). We then save these dataloaders as pickle files as we want to compare different models on the same training and test data. We utilize the GPU available to ensure we can run 20-30 epochs within only 10-15 minutes. All of the code shown in this milestone is available in the file 'transfer_learning_CNN.py'. Below, we examine further in code, how we obtained the training, validation (used for early stopping to prevent overfitting), and testing data:

```python
dataset = ImageDataset() # Initialize the class as shown before
train_loader, valid_loader, test_loader = split_dataset(dataset, 0.8, 0.1) # Use the split_dataset function - 0.8 (training), 0.1(validation)
# The split_data function:

def split_dataset(dataset, train, valid):
    train_validation = int(len(dataset) * train) # Get amount of train samples
    validation_split = int(train_validation * valid) # Get validation from train
    test_split = int(len(dataset) - train_validation) # subtract from train to get test (remaining data)
    
    # Use random split to split using the numbers
    train_data, test_data = random_split(dataset, [train_validation, test_split], generator=torch.Generator().manual_seed(100))
    train, validation = random_split(train_data, [int(train_validation - validation_split), validation_split], generator=torch.Generator().manual_seed(100))

    # put the data into dataloaders to iterate through them in batches
    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    valid_loader = DataLoader(validation, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
```
- Next we define our CNN class where we use 3 convolutional layers, having some max pooling of size 2x2 and connected with ReLU activation functions with a softmax function at the end to output probabilities of each class. The max pooling and dropout layers are present for regularisation and reduce number of parameters trained to reduce overfitting and run time. The code snippet that represents the deep learning architecture connected using torch Sequential module is shown below with a softmax function at the end to output probabilities:

```python

self.layers = torch.nn.Sequential(
    torch.nn.Conv2d(3, 200, 5, 2), # Kernel size 7 with stride 2
    torch.nn.MaxPool2d(4,4),
    torch.nn.ReLU(),

    torch.nn.Conv2d(200, 100, 3),
    # torch.nn.MaxPool2d(2, 2), # Max pooling (2, 2) filter
    torch.nn.Dropout(p=0.2),
    torch.nn.ReLU(),

    torch.nn.Conv2d(100, 50, 3),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Dropout(p=0.3),
    torch.nn.ReLU(),

    torch.nn.Flatten(), # flatten all the torch tensor to be able to output 13 features at the end
    torch.nn.Linear(1250, 100),
    torch.nn.Linear(100, 13), #  Predicting 13 product categories
    torch.nn.Softmax(dim=1)
)
```

Furthermore, apart from defining functions and classes for employing early stopping (shown below) and calculating training, validation loss and testing accuracy, we use the tensorboard library to plot graphs of the validation and training losses to visually inspect the change in loss with respect to every batch or epoch. Lastly, we do not expect a high accuracy as we only run our model for 20 epochs and to learn the patterns in our image dataset would require hours of training with multiple GPUs if we want to train a CNN from scratch. 

Here is a code snippet to show how our early stopping class is defined where it checks the average validation loss after every epoch and if it keeps increasing continually, the training function stops and returns the model:

```python

# Early stopping
class EarlyStopping():
    def __init__(self, patience=4): # patience is how many times the validation loss per epoch is allowed to increase
        self.patience = patience
        self.counter = 0
        self.early_stop = False

    def __call__(self, previous_val_loss, curr_val_loss):
        if curr_val_loss >= previous_val_loss:
            self.counter +=1 # counter keeps track
            if self.counter >= self.patience:  
                self.early_stop = True
        else:
            self.counter = 0 # if the next epoch has lower validation, counter goes back to zero
```

Shown below are code snippets from the transfer_learning_CNN.py file displaying what steps our train function goes through to train model parameters based on cross entropy loss (multiclassification) where to summarize, the training function saves the model weights after every epoch in the 'model_evaluation' folder and the best model parameters in the 'final_models' directory. The model uses the adam optimizer due it being computationally efficient, requiring less memory space (fewer parameters) and working well with large datasets. 

```python
 # Early stopping
early_stopping = EarlyStopping(patience=6)
last_loss = np.inf
# patience = 2
# triggertimes = 0

optimiser = torch.optim.SGD(model.parameters(), lr=0.009)
# weight decay adds penalty to loss function, shrinks weights during backpropagation to prevent overfitting and exploding gradients
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
        optimiser.zero_grad() # update model parameters or weights
        # exp_lr_scheduler.step()
        writer.add_scalar('Training Loss', loss.item(), batch_idx)
        loss_per_epoch += loss.item()
        batch_idx += 1

    print(f'epoch number {epoch} with average loss: {loss_per_epoch / len(train_loader)}')


    # Early stopping
    validation_loss = validation(model, device, valid_loader, F.cross_entropy) 
    early_stopping(last_loss, validation_loss)
    last_loss = validation_loss


    if early_stopping.early_stop:
        print("We are at epoch:", epoch) # stop if model overfits (validation loss continues to increase)
        return model

```
Moreover, shown below are the training and validation loss plots from tensorboard which clearly shows the trend of loss decreasing with every increasing epoch:

<p align="center">
<img src='https://user-images.githubusercontent.com/51030860/183224538-728fa7d0-b2e0-47dc-9fbc-e7559a1160b4.png'>
</p>

Results:

> Epoch number 20 with average loss: 2.5741783587012703 \
> 100%|██████████| 21/21 [05:32<00:00, 15.83s/it] \
> Accuracy of the network on the test images: 10.0 %

## Milestone 5: Create the text understanding model

## Milestone 6: Combine the models

## Milestone 7: Configure and deploy the model serving API

- Does what you have built in this milestone connect to the previous one? If so explain how. What technologies are used? Why have you used them? Have you run any commands in the terminal? If so insert them using backticks (To get syntax highlighting for code snippets add the language after the first backticks).

- Example below:

```bash
/bin/kafka-topics.sh --list --zookeeper 127.0.0.1:2181
```

- The above command is used to check whether the topic has been created successfully, once confirmed the API script is edited to send data to the created kafka topic. The docker container has an attached volume which allows editing of files to persist on the container. The result of this is below:

```python
"""Insert your code here"""
```

> Insert screenshot of what you have built working.

## Milestone n

- Continue this process for every milestone, making sure to display clear understanding of each task and the concepts behind them as well as understanding of the technologies used.

- Also don't forget to include code snippets and screenshots of the system you are building, it gives proof as well as it being an easy way to evidence your experience!

## Conclusions

- Maybe write a conclusion to the project, what you understood about it and also how you would improve it or take it further.

- Read through your documentation, do you understand everything you've written? Is everything clear and cohesive?
