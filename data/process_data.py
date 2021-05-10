import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads 2 input files from the arguments
    and converts to one dataframe.
    '''
    messages = pd.read_csv(messages_filepath, index_col = 0)
    categories = pd.read_csv(categories_filepath, index_col = 0)
#     categories = categories['categories'].str.split(';', expand=True)
    
    return pd.concat([messages, categories], axis=1)

def clean_data(df):
    '''
    Take a dataframe as input.
    Transpose 'categories' column into multiple columns.
    Cleans the categories data so the cell values contain binary.
    Output the dataframe.
    '''
    df_temp = df['categories'].str.split(';', expand=True) # Transpose the ';' separated text into multiple columns
    row = df_temp.iloc[:1, :].values.flatten() # Take the first row of the dataframe
    category_colnames = list(map(lambda x: x[:-2], row))  # Extract the string portion of each item into a list
    df_temp.columns = category_colnames  # Apply the list as column names

    for column in category_colnames:
    # set each value to be the last character of the string
        df_temp[column] = df_temp[column].str[-1]

        # convert column from string to numeric
        df_temp[column] = df_temp[column].astype(int)
    
    ## Upon reviewing the categories data, 'related' had 0, 1, 2.  
    ## For value 2, the rest of the rows were all 0's which match the pattern of value 0,
    ## which means 2 represents the same meaning as 0.
    ## In order to convert the data to binary, 2 is replaced with 0
    df_temp['related'] = df_temp['related'].replace(2,0)

    df = pd.concat([df.drop(columns='categories'), df_temp], axis=1)
        
    return df.drop_duplicates()
    
    
def save_data(df, database_filename):
    '''
    Save the input dataframe to SQLite database with the input database name
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_msg_cats', engine, index=False, if_exists='replace')  


def main():
    '''
    Cleans and transform the input data into ML ready dataframe
    And saves it to a database.
    '''
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