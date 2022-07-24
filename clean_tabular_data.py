import pandas as pd

def clean_price(price_column):

    """This function takes input a pandas series price column and removes all characters except for digits and decimal point

    Args:
        price_column (pd.Series): Price column with string format

    Returns:
        pd.Series: Price column for dataframe in float format 

    """
    return price_column.str.replace('[^0-9.]', '', regex=True).astype('float64')

def clean_time(date_col):
    """This function converts a pandas series with date information into datetime format
    
    Args:
        date_col (pd.Series): date column of a DataFrame

    Returns:
        pd.Series: Datetime column with format %D/%M/%Y
    """
    return pd.to_datetime(date_col, infer_datetime_format=True, errors='coerce').dt.strftime('%d/%m/%Y')

def get_tabular_data(filepath, line_term=','):
    """This function imports a .csv file as a pandas dataframe and drops all null rows

    Args:
        filepath (str): Path to desired csv file 
        line_term (str): Line terminator implemented in csv file where ',' is by default

    Returns:
        pd.DataFrame: The desired dataframe obtained from the csv file with all missing data removed
    
    """
    df = pd.read_csv(filepath_or_buffer=filepath, lineterminator=line_term).dropna()
    return df


def text_split(column, character):
    """The function splits the values in every row from a specificed character
    
    Args:
        column (pd.Series): The desiredcolumn of the pandas dataframe
        character (str): The character by which we split every row from

    Returns:
        pd.Series: Pandas series with all values before the character we split every row by
    """
    return column.apply(lambda x: x.split(character)[0].lower())

def clean_text_data(column, keep_char=None):
    """The function removes all non-alphanumeric and whitespace characters from the text and allows us to keep a certain number of words 
    
    Args:
        columns (pd.Series): The required text column of DataFrame
        keep_char (int): The number of words from the beginning we want to keep

    Returns:
        pd.Series: Pandas series containing only numeric and alphabetic data

    """
    non_alpha_numeric = column.str.replace('\W', ' ', regex=True).apply(lambda x: x.lower())
    non_whitespace = non_alpha_numeric.str.replace('\s+', ' ', regex=True)
    # remove all single characters
    clean_text = non_whitespace.apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))
    # Only when we specify the number of words we want to keep, we run this code
    if keep_char != None:
        return clean_text.apply(lambda x: ' '.join(x.split(' ')[0:keep_char])) 
    return clean_text


def get_data_pipeline():
    """This function acts a pipeline where all the previous functions are run to return a cleaned dataset

    Returns:
        pd.DataFrame: A clean product dataset is returned for further analysis
    """

    # Import data
    data = get_tabular_data('Products.csv', '\n').iloc[:, 1:] # We avoid the Unnamed: 0 column
    # We get duplicates with only these columns as our criteria and keep only the first occuring values
    data.drop_duplicates(subset=['product_name', 'location', 'product_description', 'create_time', 'price'], keep='first', inplace=True)

    # remove unnecessary column
    data.drop(columns=['url', 'page_id'], inplace=True)

    data['price'] = clean_price(data['price'])
    data['create_time'] = clean_time(data['create_time'])

    data['location'] = data['location'].astype('category')
    data['category'] = data['category'].astype('category')


    data['product_name'] = clean_text_data(data['product_name'], 8)
    data['product_description'] = clean_text_data(data['product_description'])

    # renmae id to product_id
    data.rename(columns={'id': 'product_id'}, inplace=True)

    return data

if __name__ == '__main__':
    get_data_pipeline()
