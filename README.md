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

- Regarding the images dataset, we also create a pipeline which resizes all the images into one consistent format such that all images have the same number of channels and size (3, 64, 64). As every product can have more than one corresponding image, we need to join the image and product tabular datasets. We merge them after cleaning the product tabular dataset on image_id and we drop all irrelevant columns such as product_id, create_time, bucket_link and image_ref. Next, we see that the images in the image folder are named by their ids. When we loop through the images in the directory, we first check that the image id is first present in our merged dataframe, then we apply our resizing image function, and lastly save these newly resized images into the cleaned_images directory where the image names are their ids. This ensures we have the same number of dimensions when performing image classification

- To summarize, we first merge our data, then we loop through the imagees in the image folder, check that the id exists in the merged dataframe, them apply our resize function, and finally save the new images by their ids in the cleaned_images folder. A code snippet is shown below of how it is done:

```python

merged_data = merge()

# check if cleaned_images exists
new_path = 'cleaned_images/'
if not os.path.exists(new_path):
    os.makedirs(new_path)

final_size = 64

for item in dirs:
    if item.split('.')[0] in list(merged_data['image_id'].unique()): # the file name of every image (image_id)
        im = Image.open(path + item)
        new_im = resize_image(final_size, im)
        new_im.save(f'{new_path}{item}')


```

  
- For resizing images, we use pillow and os libraries in python where the clean_image_data function takes in the path for the folder containing all the images, opens all the images using a for loop, resizes all of them and saves them into a new directory called cleaned_images. Below is a snippet shown of the process of how we resize all images and having only RGB channels. We only use the final size as 64 as a large pixel sizes will increase the number of predictors and the machine learning classification model time.
  
```python
size = im.size
ratio = float(final_size) / max(size) 
new_image_size = tuple([int(x*ratio) for x in size]) 
im = im.resize(new_image_size, Image.ANTIALIAS)
new_im = Image.new("RGB", (final_size, final_size))
new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))

```

## Milestone 3: Create simple Machine Learning models

We created two simple ML models using Linear regression and Logistic regression (for classifiation):

- Predicting the product price based on product name, description, and location
- Predicting product category based on the image dataset converted into numpy arrays

<ins>1 - Linear Regression model for predicting price (Regression):</ins>

First we split the data into features (name, description, location) and targets (price) to then transform our features using TfidfVectorizer where we convert all the text into weights assigned to each word based on their term frequency. Additionally, we exclude stopwords from our features such as 'the', 'are' etc. This is done to remove unnecessary words from hindering our model performance. Next we have hyperparameters we define for Gridsearch to select the optimal and then lastly we perform linear regression. We do get a terrible RMSE (approx 8000) and r^2 score (approx -0.1) as we have too many features (curse of dimensionality) and so perhaps we can focus on removing further words from our model. Furthermore, we only keep the first 8 words in the product name to avoid having a seriously long name in our analysis. 

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
    'tfidf__vector_2__min_df': (0.001, 0.005),

    'tfidf__vector_2__ngram_range': ((1, 1),(1, 2)),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2)
# split data in to train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

grid_search.fit(X_train, y_train)

rmse = np.sqrt(mean_squared_error(y_test, grid_search.predict(X_test)))
print(f'RMSE: {rmse}')
print(f'The r^2 score was: {r2_score(grid_search.predict(X_test), y_test)}')
```
<ins>2 - Logistic Regression for predicting product category (Classification):</ins>

Firstly, we obtain the images from the cleaned_images folder and convert to numpy array format and reshape them as 2D to be able to store the images as a dataframe. The total number was 12,600 where we saved the dataframe as a pickle file to prevent the array format of images being changed after we reload the dataframe. Next we sorted the merged dataframe by image id so we have the same ordering as the files in the cleaned_images folder, we perform train-test split and use logistic regression for classification for all 13 categories. We obtain around 8.5% accuracy which is poor but it gives us a benchmark to compare and improve upon when using deep learning frameworks. We print the classification report additionally which gives us the precision, recall, and f1-score for each category where we can see that our model performs best when predicting the Home & Garden category. For future, we can have greater pixel sizes for our images as much of the detail in the images with (64x64) pixels is lost. Lastly, we can exploit further hyperparameter tuning using Grid Search instead of Randomized Search, cross-validation and potentially regularization to reduce variance in the data and reduce overfitting. Shown below is the code snippet we use to run the model:

```python
df.sort_values(by='id', inplace=True) # So that the order of the images in both the tabular and non-tabular are the same
df.category = df.category.apply(lambda x: x.split('/')[0]) # Get the category most closest to the product (the one on the most left)
df.category = df.category.astype('category')
df['category_codes'] = df.category.cat.codes

y = df.category_codes # target variable
X = images_to_array('cleaned_images/')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = [    
{'penalty' : ['l2', 'none'],
'C' : np.logspace(-4, 4, 20),
'solver' : ['newton-cg','lbfgs', 'sag','saga'], # For multi-classification
'max_iter' : [300, 1000, 1500],
}
]

model = LogisticRegression()

random_search = sklearn.model_selection.RandomizedSearchCV(
    estimator=model, 
    param_distributions=param_grid

    )

random_search.fit(X_train, y_train)
y_pred = random_search.predict(X_test)

print(f'The accuracy of our predictions: {round(accuracy_score(y_test, y_pred), 5) * 100} %')
print(classification_report(y_test, y_pred))
print(dict(enumerate(df['category'].cat.categories))) # Prints which code corresponds to which category
```

## Milestone 4: Creating a pytorch vision CNN model

## Milestone 5: Using transfer learning and Resnet

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
