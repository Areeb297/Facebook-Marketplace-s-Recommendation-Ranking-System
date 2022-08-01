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


def category_encode_decode(dataframe):
    """This function returns the decoding of product categories after using cat.codes

    Args:
        dataframe (pd.DataFrame): Merged dataframe of products and image information

    Returns
        dict: Decoded dictionary for product categories
        pd.DataFrame: Updated dataframe with encoded categories
    """
    
    dataframe.category = dataframe.category.astype('category')
    dataframe['category_codes'] = dataframe.category.cat.codes

    return dict(enumerate(dataframe['category'].cat.categories)), dataframe

def image_classification():
    """This function gets the product and image tabular data merged together, converts the category into numerical codes,
    takes the category as target, splits the data into traning and testing and use logistic regression to predict the 
    product category based on the image data, printing the accuracy score and classification report
    """
    if not os.path.exists('image_dataframe.pkl'):
        merge_data.merge_im_array()

    file = open("image_dataframe.pkl",'rb')
    df = pickle.load(file)

    df.category = df.category.apply(lambda x: x.split('/')[0]) # Retain only the first category

    decode, df = category_encode_decode(df)
    with open('decoder.pkl', 'wb') as file:
        pickle.dump(decode, file)

    y = df.category_codes # target variable
    with open('image_targets.pkl', 'wb') as file:
        pickle.dump(y, file)

    X = df['image_array'].apply(lambda x: x.flatten())

    with open('features.pkl', 'wb') as file:
        pickle.dump(X, file)

    X_train, X_test, y_train, y_test = train_test_split(list(X), y, test_size=0.3, random_state=42)

    param_grid = [    
    {'penalty' : ['l2'],
    'C' : np.logspace(-4, 4, 30),
    'solver' : ['lbfgs'], # For multi-classification (newton-cg, sag, saga)
    'max_iter' : [300, 600, 1200],

    }
    ]

    model = LogisticRegression()

    random_search = sklearn.model_selection.RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_grid,
        verbose=4, n_iter=2, cv=3
        
        )


# For small datasets, liblinear is a good choice, whereas sag and saga are faster for large ones;
# For multiclass problems, only newton-cg, sag, saga and lbfgs handle multinomial loss;
# liblinear is limited to one-versus-rest schemes.

    
    random_search.fit(X_train, y_train)
    y_pred = random_search.predict(X_test)

    print(random_search.best_params_)
    print(f'The accuracy of our predictions: {round(accuracy_score(y_test, y_pred), 5) * 100} %')
    print(classification_report(y_test, y_pred))
    print(decode) # Prints which code corresponds to which category

if __name__ == '__main__':
    image_classification()
