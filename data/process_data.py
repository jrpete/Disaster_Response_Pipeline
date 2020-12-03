import sys
import pandas as pd
import sqlalchemy 

def load_data(messages_filepath, categories_filepath):
    """loads data from messages and categories csv files
    Args:
        messages_filepath (string): The path to the disaster_messages.csv file
        categories_filepath (string): The path to the disaster_categories.csv file
    Returns:
        df (data frame): A dataframe containing the combined messages and categories data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return(messages, categories)

def clean_data(df):
    """Merges, drops duplicates, and outputs a cleaned version of the combined messages and categories data frame
    Args:
        df (data frame):  A dataframe containing the combined messages and categories data
    Returns:
        df (data frame): A new dataframe containing a cleaned version of the combined messages and categories data

    """
    # merge datasets
    df = df[0].merge(df[1],left_on='id', right_on='id')
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = lambda x: x.str.slice(0,-2)

    # rename the columns of `categories`
    categories.columns = category_colnames(row)
    category_nums = lambda x: x.str.slice(-1)

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = category_nums(categories[column]).astype(str)
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from `df`
    df = df.drop(columns = ['categories'])

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)

    # check number of duplicates
    duplicated_df = df.duplicated()

    # drop duplicates
    df.drop_duplicates(subset='message', inplace=True)

    # check number of duplicates
    df.size

    return(df)


def save_data(df, database_filename):
    """Output the result to a database
    Args:
        df (data frame): The dataframe containing a cleaned version of the combined messages and categories data
        database_filename(string): The path to the data base
    Returns:
        None
    """
    engine = sqlalchemy.create_engine('sqlite:///'+database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')
    engine.dispose()

def main():
    """
    Calls each function in the process_data.py module
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print(df)
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()