import numpy as np

import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def image_classification():
    """This function gets the merged product and image tabular dataset from pickle file, takes the image array column as features
    and category codes as targets, splits the data into traning and testing and implements logistic regression to predict the 
    product category based on the image data, printing the accuracy score and classification report.
    """
    df = pd.read_pickle('ML_product_images.pkl')
    decode = pd.read_pickle('decoder.pkl')

    X = df['image_array'].apply(lambda x: x.flatten()) # Flatten array so every row contains one flattened array
    y = df.category_codes

    X_train, X_test, y_train, y_test = train_test_split(list(X), y, test_size=0.3, random_state=42)

    param_grid = [    
    {'penalty' : ['l2'],
    'C' : np.logspace(-4, 4, 30),
    'solver' : ['lbfgs'], # For multi-classification (newton-cg, sag, saga)
    'max_iter' : [400],

    }
    ]

    model = LogisticRegression()

    random_search = sklearn.model_selection.RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_grid,
        verbose=4, n_iter=2, cv=2
        
        )


    # For small datasets, liblinear is a good choice, whereas sag and saga are faster for large ones;
    # For multiclass problems, only newton-cg, sag, saga and lbfgs handle multinomial loss;
    # liblinear is limited to one-versus-rest schemes.

    # We have 30x30 images --> 900 featues

    
    random_search.fit(X_train, y_train)
    y_pred = random_search.predict(X_test)

    print(random_search.best_params_)
    print(f'The accuracy of our predictions: {round(accuracy_score(y_test, y_pred), 5) * 100} %')
    print(classification_report(y_test, y_pred))
    print(decode) # Prints which code corresponds to which category

if __name__ == '__main__':
    image_classification()


# Results from one run:

# {'solver': 'lbfgs', 'penalty': 'l2', 'max_iter': 400, 'C': 0.03039195382313198}
# The accuracy of our predictions: 15.201999999999998 %
#               precision    recall  f1-score   support

#            0       0.10      0.11      0.11       228
#            1       0.06      0.07      0.06       167
#            2       0.16      0.14      0.15       224
#            3       0.21      0.17      0.19       315
#            4       0.13      0.12      0.12       230
#            5       0.15      0.15      0.15       308
#            6       0.19      0.21      0.20       368
#            7       0.17      0.20      0.18       266
#            8       0.19      0.20      0.19       300
#            9       0.16      0.13      0.14       246
#           10       0.16      0.15      0.16       226
#           11       0.12      0.13      0.13       231
#           12       0.11      0.10      0.11       226

#     accuracy                           0.15      3335
#    macro avg       0.15      0.15      0.15      3335
# weighted avg       0.15      0.15      0.15      3335

# {0: 'Appliances ', 1: 'Baby & Kids Stuff ', 2: 'Clothes, Footwear & Accessories ', 3: 'Computers & Software ', 4: 
# 'DIY Tools & Materials ', 5: 'Health & Beauty ', 6: 'Home & Garden ', 7: 'Music, Films, Books & Games ', 8: 
# 'Office Furniture & Equipment ', 9: 'Other Goods ', 10: 'Phones, Mobile Phones & Telecoms ', 11: 'Sports, Leisure & Travel ', 12: 'Video Games & Consoles '}
