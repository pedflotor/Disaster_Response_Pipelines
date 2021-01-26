import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import TruncatedSVD
import pickle
import sqlite3
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
    Function to load the necessary data from the cleaned database created with the process_data.py script.
    The features (x), target variables (y) and category names (category_names) will be returned

    Parameters
    ----------
    database_filepath: str
        path of the database file with the cleaned data

    Returns
    -------
    X:object
        dataframe with the features
    Y:object
        dataframe with the target variables
    category_names:object
        dataframe with the target names

    """

    engine = create_engine('sqlite:///' + database_filepath)
    query = 'SELECT * FROM DisasterResponse;'
    df = pd.read_sql_query(query, engine)

    x = df["message"]
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns

    return x, y, category_names


def tokenize(text):
    """
    Function to process the text data

    Parameters
    ----------
    text:object
        text data with the raw messages that will be processed

    Returns
    -------
    clean_tokens:object
        text data tokenized, processed and ready to be used for the machine learning algorithm
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
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