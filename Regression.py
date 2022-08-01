import clean_tabular_data
from nltk.corpus import stopwords
import numpy as np


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



def regression_func():
    """This function gathers the required product data, performs TF-IDF on the text data 
    and prints the RMSE and R^2 score
    
    """
    # We get the cleaned product tabular data from the clean_tabular_data python script
    product_data = clean_tabular_data.get_data_pipeline()

    # Features
    X = product_data[['product_name', 'product_description', 'location']]
    y = product_data['price']

    # stopwords list to exclude 
    stop = set(stopwords.words('english'))

    tfidf = ColumnTransformer(  

    [
        ("vector_1", TfidfVectorizer(stop_words=stop), 'product_name'),
        ("vector_2", TfidfVectorizer(stop_words=stop), 'product_description'),
        ("vector_3", TfidfVectorizer(), 'location')], remainder='passthrough') # We don't need stopwords for location



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

    # Find the best parameters for both the feature extraction and regressor
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2)


    # split data in to train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    grid_search.fit(X_train, y_train)

    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, grid_search.predict(X_test)))
    print(f'RMSE: {rmse}')
    print(f'The r^2 score was: {r2_score(grid_search.predict(X_test), y_test)}')


if __name__ == '__main__':
    regression_func()
