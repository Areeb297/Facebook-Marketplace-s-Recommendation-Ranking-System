import merge_data
import numpy as np
import os

from PIL import Image
import pandas as pd
import pickle

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def images_to_array(path):
    """This function converts all the image files in cleaned_images to np.array format, reshapes them into 2D array,
    stores them as a list, converts the data into a pandas DataFrame and saves the dataframe as a pickle file

    Args:
        path (str): The path to where the cleaned images are kept
    Returns:
        pd.DataFrame: The dataframe with images as arrays
    """
    # check first if pickle file already exists
    if not os.path.exists('image_dataframe.pkl'):
        arr_im_list = []
        dirs = os.listdir(path)

        for item in dirs:
            im = Image.open(path + item)
            numpydata = np.asarray(im) # convert image to array
            arr_im_list.append(numpydata)

        featueres = pd.DataFrame(np.reshape(arr_im_list, (12600, 64*64*3))) # (64*64*3) are the sizes we saved the images as 
            # Store data (serialize)
        with open('image_dataframe.pkl', 'wb') as file:
            pickle.dump(featueres, file)
    else:   
        file = open("image_dataframe.pkl",'rb')
        featueres = pickle.load(file)
    
    return featueres

def image_classification():
    """This function gets the product and image tabular data merged together, converts the category into numerical codes,
    takes the category as target, splits the data into traning and testing and use logistic regression to predict the 
    product category based on the image data, printing the accuracy score and classification report
    """

    df = merge_data.merge()
    df.sort_values(by='id', inplace=True) # So that the order of the images in both the tabular and non-tabular are the same
    df.category = df.category.apply(lambda x: x.split('/')[0])
    df.category = df.category.astype('category')
    df['category_codes'] = df.category.cat.codes

    y = df.category_codes # target variable
    with open('image_targets.pkl', 'wb') as file:
        pickle.dump(y, file)

    X = images_to_array('cleaned_images/')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
    ]

    model = LogisticRegression()

    random_search = sklearn.model_selection.RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_grid
    
        )
    
    random_search.fit(X_train, y_train)
    y_pred = random_search.predict(X_test)

    print(random_search.best_params_)
    print(f'The accuracy of our predictions: {round(accuracy_score(y_test, y_pred), 5) * 100} %')
    print(classification_report(y_test, y_pred))
    print(dict(enumerate(df['category'].cat.categories))) # Prints which code corresponds to which category

if __name__ == '__main__':
    image_classification()
