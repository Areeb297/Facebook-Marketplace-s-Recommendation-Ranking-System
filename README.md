# FB Marketplace Recommendation Ranking System

## Milestone 1: An overview of the system

> This project works on building and replication a product ranking system of Facebook Marketplace to provide users with the most relevant products based on their search query through using using multimodal pre-traineddeep neural networks such as CNNs using Transfer Learning. The model is a mini implementation of a larger system that Facebook developed and advanced to generate product recommendation rankings for buyers on Facebook Marketplace. Shown below is a flowchart describing the overview of the system encompassing various technologies:

![image](https://user-images.githubusercontent.com/51030860/178149528-8a7c5b0c-3f14-46b0-b708-ff3faf455755.png)

Here is the ![video link](https://www.youtube.com/watch?v=1Z5V2VrHTTA&ab_channel=AiCore) for further information and reference.

## Milestone 2: Cleaning the tabular and image datasets

- In this stage, we perform data cleaning steps for the product and image datasets. Firstly, conncerning the product tabular dataset, we have a pipeline which completes all cleaning steps such as ensuring all null and duplicate values are removed and all data formats are correct e.g., the price column is converted to float and the product creation time is converted to datetime. Regarding the images dataset, we create a pipeline which resizes all the images into one consistent format such that all images have the same number of channels and size. An important point to mention is that the images in the folder are named by their id so we will sort these images first and then apply the image resizing pipeline. Similarly, when we merge both the image and product tabular datasets, we will sort the dataframe by image id which will help us in the classification task.

- Moreover, in order to have only the images that are in the image tabular dataset, first we merge the product and image datasets together, perform all the cleaning steps and then before resize the image, we check whether the id (name of the image file) is in the unique image id's in the tabular dataset. This ensures we have the same number of dimensions when performing image classification. Below is a code snippet that shows how we do it:

```python
# check if cleaned_images exists
new_path = 'cleaned_images/'
if not os.path.exists(path+new_path):
    os.makedirs(new_path)

final_size = 90

for n, item in enumerate(dirs, 1):
    if item.split('.')[0] in list(images_data['id'].unique()): # Here we check whether the image id is contained in the merged tabular dataset
        im = Image.open(path + item)
        new_im = resize_image(final_size, im)
        new_im.save(f'{new_path}{n}_resized.jpg')
```

- We use the pandas library and Regex expression to clean the product dataset for example, we only keep alphanumeric characters and remove uncessary spaces as shown below:

```python

non_alpha_numeric = column.str.replace('\W', ' ', regex=True).apply(lambda x: x.lower())
non_whitespace = non_alpha_numeric.str.replace('\s+', ' ', regex=True)
# remove all single characters
clean_text = non_whitespace.apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))
```
  
- For images, we use pillow and os libraries in python where the clean_image_data function takes in the path for the folder containing all the images, opens all the images using a for loop, resizes all of them and saves them into a new directory called cleaned_images. Below is a snippet shown of the process of how we resize all images and having only RGB channels. We only use the final size as 90 as large pixel sizes will increase the machine learning classification model time.
  
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

First we split the data into features (name, description, location) and targets (price) to then transform our features using TfidfVectorizer where we convert all the text into weights assigned to each word based on their term frequency. Additionally, we exclude stopwords from our features such as 'the', 'are' etc. This is done to remove unnecessary words from hindering our model performance. Next we have hyperparameters we define for Gridsearch to select the optimal and then lastly we perform linear regression. We do get a terrible RMSE (~8000) and r^2 score (~ -0.1) as we have too many features (curse of dimensionality) and so perhaps we can focus on removing further words from our model. Furthermore, we only keep the first 8 words in the product name to avoid having a seriously long name in our analysis. 

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

Firstly, we obtain the images from the cleaned_images folder and convert to numpy array format and reshape them as 2D to be able to store the images as a dataframe. The total number was 12,600 where we saved the dataframe as a pickle file to prevent the array format of images being changed after we reload the dataframe. Next we sorted the merged dataframe by image id so we have the same ordering as the files in the cleaned_images folder, we perform train-test split and use logistic regression for classification for all 13 categories. We obtain around 7.5% accuracy which is poor but it gives us a benchmark to compare and improve upon when using deep learning frameworks. We print the classification report additionally which gives us the precision, recall, and f1-score for each category where we can see that our model performs best when predicting the video games & consoles category. For future, we can have greater pixel sizes for our images as much of the detail in the images with (90x90) pixels is lost. Lastly, we can exploit hyperparameter tuning, cross-validation and potentially regularization to reduce variance in the data and reduce overfitting. Shown below is the code snippet we use to run the model:

```python
df.sort_values(by='id', inplace=True) # So that the order of the images in both the tabular and non-tabular are the same
df.category = df.category.apply(lambda x: x.split('/')[0]) # Get the category most closest to the product (the one on the most left)
df.category = df.category.astype('category')
df['category_codes'] = df.category.cat.codes

y = df.category_codes # target variable
X = images_to_array('cleaned_images/')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'The accuracy of our predictions: {round(accuracy_score(y_test, y_pred), 5) * 100} %')
print(classification_report(y_test, y_pred))
print(dict(enumerate(df['category'].cat.categories))) # Prints which code corresponds to which category
```

## Milestone 4: Create the vision model

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
