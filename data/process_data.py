import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath, index_col = 0)
    categories = pd.read_csv(categories_filepath, index_col = 0)
    
    return pd.concat([messages, categories], axis=1)

def clean_data(df):
    '''
    Expand the categories to each column.  
    Apply the category names as column names
    And only leave numeric values in the cells.
    Remove the original categories column 
    And append the new expanded category columns
    To the message columns.
    Return the new dataframe
    '''
    df_temp = df['categories'].str.split(';', expand=True)
    row = df_temp.iloc[:1, :].values.flatten()
    category_colnames = list(map(lambda x: x[:-2], row))
    df_temp.columns = category_colnames

    for column in category_colnames:
    # set each value to be the last character of the string
        df_temp[column] = df_temp[column].str[-1]

        # convert column from string to numeric
        df_temp[column] = df_temp[column].astype(int)
        
    df = pd.concat([df.drop(columns='categories'), df_temp], axis=1)
        
    return df.drop_duplicates()
    
    
def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_msg_cats', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

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