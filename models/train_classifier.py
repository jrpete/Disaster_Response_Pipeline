import sys

import joblib
import nltk
import pandas as pd
from sqlalchemy import create_engine
import numpy as np 

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def load_data(database_filepath):
    """Loads Messages data into X and Y variables from the DisasterResponse database
    Args:
        database_filepath (string): The filepath to the database
    Returns:
        X (data frame): The original message data
        Y (data frame): The category labels to be predicted
        category_names (list): The category names for the labels

    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    engine.dispose()

    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    category_names = Y.columns.tolist()

    return X,Y,category_names

def tokenize(text):
    """Tokenizes then lemmitizes words
    Args:
        text (string): The messages to be tokenized
    Returns:
        tokens (list): The cleaned messages in tokenized messages
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """Uses a Gridsearch Pipeline to train the classifier
    Args:
        None
    Returns:
        cv (GridSearchCV): Grid search model object

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #takes 1-2 hours to run using these params
    parameters = {'clf__estimator__n_estimators': [50,100],

        'clf__estimator__min_samples_split': [2, 4]
    }
    # Define Pipeline
    cv_pipeline = GridSearchCV(pipeline, param_grid=parameters, cv=None, verbose=12, n_jobs=-1)


    return cv_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Outputs the classification results
    Args:
        model (dataframe): The MultiOutput Classification Model 
        X_test (dataframe): The set of test messages
        Y_test (dataframe): The set of test labels 
        category_names (list): The test label column names
    Returns:
        None
    """
    Y_pred = model.predict(X_test)

    print(classification_report(np.hstack(Y_test.values),np.hstack(Y_pred)))

def save_model(model, model_filepath):
    """Outputs the model to a specified filepath
    Args:
        model (Scikit-learn Model): The trained and fitted model
        model_filepath (string): The path to output the model
    Returns:
        None
    """
    joblib.dump(model, model_filepath)


def main():
    """
    Calls each function in the train_classifier.py module
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
   

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