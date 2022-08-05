# FB Marketplace Recommendation Ranking System

## Milestone 1: An overview of the system

> This project works on building and closely replicating a product ranking system of Facebook Marketplace to provide users with the most relevant products based on their search query. This is done via a multimodal pre-trained deep neural network using Transfer Learning (Text model + Image model). The model is a mini-implementation of a larger system that Facebook developed to generate product recommendation rankings for buyers on Facebook Marketplace. Shown below is a flowchart describing the overview of the system encompassing various technologies:

![image](https://user-images.githubusercontent.com/51030860/178149528-8a7c5b0c-3f14-46b0-b708-ff3faf455755.png)

Here is the ![video link](https://www.youtube.com/watch?v=1Z5V2VrHTTA&ab_channel=AiCore) for further information and reference.

## Milestone 2: Cleaning the tabular and image datasets

- In this stage, we perform data cleaning steps for the product and image datasets. Firstly, concerning the product tabular dataset, we have built a pipeline which completes all cleaning steps such as ensuring all null and duplicate values are removed and all data formats are correct e.g., the price column is converted to float and the product creation time is converted to datetime. We ensure features like location and product category are converted into 'category' format. Additionally, we clean the text data by removing non-alphanumeric characters and unnecessary whitespaces. We use the pandas library and Regex expression to clean the product dataset as shown below:

```python


def clean_text_data(column, keep_char=None):
    non_alpha_numeric = column.str.replace('\W', ' ', regex=True).apply(lambda x: x.lower())
    non_whitespace = non_alpha_numeric.str.replace('\s+', ' ', regex=True)
    # remove all single characters
    clean_text = non_whitespace.apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))
    if keep_char != None:
        return clean_text.apply(lambda x: ' '.join(x.split(' ')[0:keep_char])) # keep a certain number of words
    return clean_text

# We get duplicates with only these columns as our criteria and keep only the first occuring values
data.drop_duplicates(subset=['product_name', 'location', 'product_description', 'create_time', 'price'], keep='first', inplace=True)

# remove unnecessary columns
data.drop(columns=['url', 'page_id'], inplace=True)

data['price'] = clean_price(data['price'])
data['create_time'] = clean_time(data['create_time'])

data['location'] = data['location'].astype('category')
data['category'] = data['category'].astype('category')


data['product_name'] = clean_text_data(data['product_name'], 8)
data['product_description'] = clean_text_data(data['product_description']

```

- Regarding the images dataset, we also create a pipeline which resizes all the images into one consistent format such that all images have the same number of channels and size (3, 50, 50) for the sklearn dataset and for the CNN dataset, we obtain images of size (3, 155, 155) as this was based on finding minimum height and width from the image dataset and using the lowest number from them. 

As every product can have more than one corresponding image, we need to join the image and product tabular datasets. We merge them after cleaning the product tabular dataset on image_id and we drop all irrelevant columns such as product_id, create_time, bucket_link and image_ref.

- To build the cleaned images folder, we loop through the images in the image folder, apply our resize function by adding the image onto a rgb background of pixel size 50 and resizing our image, and finally save the new images by their ids in the cleaned_images folder. For building the CNN image dataset, we use a different approach where we find the minimum size of all the images and change the aspect ratio to that (155x155) and then we ensure the channel is RGB. A code snippet is shown below of how it is done:

```python

# For the scikit learn 50x50 dataset, we use:

def resize_image_ML(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size) 
    new_image_size = tuple([int(x*ratio) for x in size]) 
    im = im.resize(new_image_size, Image.LANCZOS)
    new_im = Image.new("RGB", (final_size, final_size)) # new_img_size = (int(ratio * prev_size[0]), int(ratio * prev_size[1]))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im


def clean_image_data_sklearn(path):
    list_img = glob.glob('images/*.jpg') # Get all files with file type .jpg
    # check if cleaned_images exists
    new_path = path
    if not os.path.exists(new_path):
        os.makedirs(new_path)

        final_size = 50 # We will use 50x50 images when using Sklearn ML libraries

        for item in list_img:
            im = Image.open(item)
            new_im = resize_image_ML(final_size, im)
            file_name = item.split('\\')[1]
            new_im.save(f'{new_path}{file_name}')
            
            
# For the CNN dataset, we use:
min_width = 100000
min_height = 100000

for img in list_img:

    size = Image.open(img).size
    width, height = size[0], size[1]

    if width < min_width:
        min_width = width

    if height < min_height:
        min_height = height

min_height = min(min_height, min_width)
min_width =  min_height

# check rgb
for img in list_img:

    new_img = Image.open(img)
    # check RGB color/mode
    if new_img.mode != 'RGB':
        new_img = new_img.convert('RGB')

```

  
- For resizing the sklearn images, we use pillow and os libraries in python where the clean_image_data function opens all the images using a for loop, resizes all of them and saves them into a new directory called cleaned_images_sklearn. Below is a snippet shown of the process of how we resize all images and having only RGB channels. We only use the final size as 50 as a larger pixel size will increase the number of predictors and the sklearn machine learning classification model time by a huge amount as it will too much memory. For the CNN, we can use large image sizes.
  
```python
final_size = 50
size = im.size
ratio = float(final_size) / max(size) 
new_image_size = tuple([int(x*ratio) for x in size]) 
im = im.resize(new_image_size, Image.ANTIALIAS)
new_im = Image.new("RGB", (final_size, final_size))
new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))

```

## Milestone 3: Create simple Machine Learning models

We created two simple ML models using Linear regression (Regression) and Logistic regression (for multi-class classifiation):

- Predicting the product price based on product name, description, and location (Linear Regression)
- Predicting product category based on the image dataset converted into numpy arrays (Logistic Regression)

<ins>1 - Linear Regression model for predicting price (Regression):</ins>

- First we split the data into features (name, description, location) and targets (price) to then transforming our features using TfidfVectorizer. We convert all the text into weights assigned to each word based on their term frequency in the whole merged dataframe. Additionally, we exclude stopwords from our features such as 'the', 'are' etc. We do not apply this transformation on the location feature as it is not needed. This process is done to remove unnecessary words from hindering our model performance. 

- Next we have hyperparameters we define for Gridsearch to select the optimal such as n_gram range and minimum term frequency. Lastly we perform linear regression. We do get a terrible RMSE (approx 230,000) and r^2 score (-0.005) as we have too many features (curse of dimensionality) and have overparametrized our model. We can potentially focus on removing further words from our model or obtain more data in the future. We can try other models like random forest regressor but they take a long time with so many features and hence may not be feasible at the moment. Furthermore, we only keep the first 8 words in the product name to avoid having a seriously long product name in our analysis. 

```python
pipeline = Pipeline(

    [   
        ("tfidf", tfidf),
        ("lr", LinearRegression())
    ]
)

# set parameters for the tfidf vectors
parameters = {
    'tfidf__vector_1__ngram_range': ((1, 1), (1, 2)),
    'tfidf__vector_2__min_df': (0.005,  0.2, 0.01),

    'tfidf__vector_2__ngram_range': ((1,1), (1,2)),
    'tfidf__vector_1__min_df': (0.2, 0.05, 0.001)
}

# Find the best hyperparameters for both the feature extraction and regressor
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2)

# split data in to train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
grid_search.fit(X_train, y_train)

# calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, grid_search.predict(X_test)))
print(f'RMSE: {rmse}')
print(f'The r^2 score was: {r2_score(y_test, grid_search.predict(X_test))}')
```
<ins>2 - Logistic Regression for predicting product category (Classification):</ins>

- Firstly, in our merge_data.py file, we create another function which loops through the images from the cleaned_images folder, converts them into numpy array format, takes the image id from the image file, checks which row of our merged dataframe corresponds to that image id, then places each image_array as a list in the correct row under the column 'image_array'. 

- The total number of observations is 12,600 where we save this dataframe as a pickle file to prevent the array format of images being changed after we reload the dataframe. Next, we load this pickle file in our image_classification python file, encode our categories into numbers, save the encoding, take the image_array column as our features where we flatten each row, take the encoded categories as our targets, perform train-test split and use logistic regression for classification for all 13 categories as shown below:

```python
file = open("image_dataframe.pkl",'rb')
df = pickle.load(file)

df.category = df.category.apply(lambda x: x.split('/')[0]) # Retain only the first category

decode, df = category_encode_decode(df)

y = df.category_codes # target variable
X = df['image_array'].apply(lambda x: x.flatten())

X_train, X_test, y_train, y_test = train_test_split(list(X), y, test_size=0.3, random_state=42)

```

 - We use grid search to optimize hyperparameters of the logistic regression function such as the max iterations, regularization (C hyperparameter) etc. We use the lbfgs solver as it is suitable for multiclassification. We did not use other solvers as they take a lot of time to run. Instead of Grid Search, we exploit randomized search to save time. All of this setup is shown below in code:

```python
  param_grid = [    
    {'penalty' : ['l2'],
    'C' : np.logspace(-4, 4, 30),
    'solver' : ['lbfgs'], # For multi-classification (newton-cg, sag, saga, lbfgs)
    'max_iter' : [300, 400]
    }
    ]

model = LogisticRegression()

random_search = sklearn.model_selection.RandomizedSearchCV(
    estimator=model, 
    param_distributions=param_grid,
    verbose=4, n_iter=2, cv=2

    )
```
 
- The results are that we obtain an accuracy of around 15% which is poor but it is better than random guessing (1/13 ~ 7% accuracy) and gives us a benchmark to compare and improve upon when using deep learning frameworks. The sklearn models are not designed to classify images so we expected poor performance. We see as shown below that Home & Garden category has the most images so it would make sense that the model performs the best on that category. We see that our image dataset is mostly balanced which indicates no issues regarding imbalanced set of classes when it comes to multiclassification.

<p align="center">
  <img src="https://user-images.githubusercontent.com/51030860/182227206-cddce56e-0503-47f0-9bd6-007cd2e7d91d.png" alt="Sublime's custom image"/>
</p>

-  We print the classification report additionally which gives us the precision, recall, and f1-score for each category where we can see that our model performs more confidently when predicting the Home & Garden category, 'Computers & Software' and 'Office Furniture & Equipment'. 
<p align="center">
<img src="https://user-images.githubusercontent.com/51030860/182907981-aa44e5e4-edea-463e-8cb1-487aaa0bd888.png" alt="Sublime's custom image"/>
</p>

- For future, we can can exploit further hyperparameter tuning using Grid Search instead of Randomized Search and try other hyperparameters if we want to experiment further with sklearn. Additionally, we can also try other classification algorithms such as XGBoost or Random Forests. Shown below is the code snippet we use to run the model:

```python
random_search.fit(X_train, y_train)
y_pred = random_search.predict(X_test)

print(random_search.best_params_)
print(f'The accuracy of our predictions: {round(accuracy_score(y_test, y_pred), 5) * 100} %')
print(classification_report(y_test, y_pred))

```

- The results are as follows:
> Best parameters: {'solver': 'lbfgs', 'penalty': 'l2', 'max_iter': 400, 'C': 2.592943797404667}, the accuracy of our predictions: 14.74 %

## Milestone 4: Creating a basic pytorch vision CNN model

- After testing with sklearn logistic regression, we use a CNN deep neural network to see our performance on the images we saved in higher pixel dimension (3x155x155). We use 3 convolutional layers, having some max pooling of size 2x2 and connected with ReLU activation functions with a softmax function at the end to output probabilities of each class. The CNN class is shown below:

```python
random_search.fit(X_train, y_train)
y_pred = random_search.predict(X_test)

print(random_search.best_params_)
print(f'The accuracy of our predictions: {round(accuracy_score(y_test, y_pred), 5) * 100} %')
print(classification_report(y_test, y_pred))

```python

class CNN(torch.nn.Module):
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
```
To obtain the relevant data, we read in the CNN merged dataframe, split the image array and category columns as features and targets respectively after flattening the image array column, and we then create a Product Dataset class which converts the features and labels into tensor format, normalizes the 3 colour channels, and outputs features and targets as a tuple. We split the data into training, validation, and testing where validation is used for early stopping in order to prevent overfitting of the data. Additionally, apart from defining functions and classes for employing early stopping and calculating training, validation loss and testing accuracy, we use the tensorboard library to plot graphs of the validation and training losses to visually inspect the change in loss with respect to every batch or epoch. Lastly, we do not expect a high accuracy as we only run our model for 20 epochs and to learn the patterns in our image dataset would require hours of training with multiple GPUs. 

Thus, after running the model, we only achieve an accuracy of 10% on the testing set which is extremely poor. The solution to this problem would be to utilize transfer learning using a pretrained model such as Resnet50 where more on this will be discussed in the next milestone. Shown below are code snippets from the basic_CNN_classification.py file displaying what steps our train function goes through to train model parameters based on cross entropy loss (multiclassification):

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
<img src='https://user-images.githubusercontent.com/51030860/183224428-2679988f-8885-4939-acbd-07149619f165.png'>
</p>


## Milestone 5: Implementing transfer learning and Resnet50

## Milestone 6: Create the text understanding model

## Milestone 7: Combine the models

## Milestone 8: Configure and deploy the model serving API

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
