import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import pickle
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
    x:object
        dataframe with the features
    y:object
        dataframe with the target variables
    category_names:object
        dataframe with the target names

    """

    engine = create_engine('sqlite:///' + database_filepath)
    query = 'SELECT * FROM DisasterResponse;'
    df = pd.read_sql_query(query, engine)

    x = df["message"]
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    return x, y, y.columns


def tokenize(text):
    """
    Function to case normalize, lemmatize, and tokenize text using nltk
    This function is used then in the machine learning pipeline to vectorize and then apply TF-IDF to the text

    Parameters
    ----------
    text:object
        text data with the raw messages that will be processed

    Returns
    -------
    clean_tokens:object
        text data tokenized, processed and ready to be used for the machine learning pipeline
    """

    # tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Function to builds a pipeline that processes text and then performs multi-output classification on the 36
    categories in the dataset. GridSearchCV is used to find the best parameters for the model.

    Parameters
    ----------

    Returns
    -------
    cv:object
        machine learning model to be trained using GridSearchCV
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {'tfidf__use_idf': [True, False],
                   'clf__estimator__n_estimators': [100, 200, 300],
                   'clf__estimator__learning_rate': [0.8, 1]}

    cv = GridSearchCV(pipeline, param_grid=parameters, return_train_score=True, verbose=2, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to Prints the f1 score, precision and recall for the test set for each category

    Parameters
    ----------
    model:object
        machine learning model build with the build_model function
    X_test:object
        dataframe with the features
    Y_test:object
        dataframe with the target variables
    category_names:object
        dataframe with the target names

    Returns
    -------
    Prints the f1 score, precision and recall for the test set for each category
    The accuracy for each category is also printed
    """

    Y_pred = model.predict(X_test)

    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))

    print("Accuracy scores for each category\n")
    for i in range(36):
        print("Accuracy score for " + Y_test.columns[i], accuracy_score(Y_test.values[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
        Function to store the classifier into a pickle file to the specified model file path.

        Parameters
        ----------
        model:object
            machine learning model build with the build_model function
        X_test:object
            dataframe with the features
        model_filepath:str
            path of the pickle file where the classifier will be stored
        Returns
        -------
        Pickle file
        """

    with open('disaster_model.pkl', 'wb') as model_filepath:
        pickle.dump(model, model_filepath)


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