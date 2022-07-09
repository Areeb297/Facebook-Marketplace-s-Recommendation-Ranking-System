import pandas as pd
import re


def replace_signs(val):
    return re.sub('[£,]', '', val)


def clean_data():

    # Import data
    data = pd.read_csv("products.csv", lineterminator="\n").drop(columns='Unnamed: 0')

    # We drop all missing entries
    data.dropna(inplace=True)
    
    # We drop duplicates
    data.drop_duplicates(inplace=True)

    # After importing the data, we remove the price symbol and comma from the price column
    data['price'] = data['price'].apply(replace_signs)
    data['price'] = data['price'].astype('float64')

    # We convert the time feature to datatime
    data['create_time'] = pd.to_datetime(data['create_time'], infer_datetime_format=True, errors='coerce')

    # We export the cleaned data as cleaned_products.csv
    data.to_csv('cleaned_products.csv')



if __name__ == '__main__':
    clean_data()

