import sys
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

# import libraries
from sqlalchemy import create_engine
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import re
import pickle

def load_data(database_filepath):
    '''
    Loads data from the input database
    Separates the variables from the labels (X=variables, Y=label)
    Converts to the arrays
    Outputs the arrays and the columns names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_msg_cats', engine)
    X = df.iloc[:, :1].values.flatten()
    Y = df.iloc[:, 3:].values
    return X, Y, df.iloc[:, 3:].columns.tolist()


def tokenize(text):
    '''
    Process Text by normalizing, tokenizing, lemmatizing
    And outputs as clean tokens.
    '''
    stop_words = stopwords.words("english")
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    # lemmatize, normalize case, and remove leading/trailing white space
    clean_tokens = [lemmatizer.lemmatize(tok.strip()) for tok in tokens if tok not in stop_words]

    return clean_tokens


def build_model():
    '''
    Building multi-output classification model
    Using a pipeline
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), 
                 ('tfidf', TfidfTransformer()),
                 ('clf', MultiOutputClassifier(DecisionTreeClassifier()))])
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluating the model using the test data
    And displays the score for each category
    '''
    y_pred = model.predict(X_test)
        
    for i in range(len(category_names)):
        lines = classification_report(Y_test[:,i], y_pred[:,i]).split('\n')
        print('{}:'.format(category_names[i]))
        for line in lines:
            print(line)


def save_model(model, model_filepath):
    '''
    Save the data in pickle file.
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    Multi-output classifier training script
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        criterions = ['gini', 'entropy']
        parameters = dict(clf__estimator__criterion=criterions)
        model = GridSearchCV(model, param_grid=parameters)
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()