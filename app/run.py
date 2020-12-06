import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """Tokenizes then lemmitizes words
    Args:
        text (string): The messages to be tokenized
    Returns:
        clean_tokens (list): The cleaned messages in tokenized messages
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("models/classifier.pkl")



@app.route('/')
@app.route('/index')
def index():
    """index webpage displays visuals and receives user input text for model
    Args:
        None
    Returns:
        None
    """
    # extract the counts and values from the genre column
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #Extract the Names, Sums, Averages, and Standard Deviations from the individual message columns
    y = df.iloc[:,4:].values
    message_sums = y.sum(axis=0)
    message_avgs = np.average(y,axis=0)
    message_stds = np.std(y,axis=0)
    category_names = df.drop(['message', 'genre', 'id', 'original'], axis=1).columns.tolist()

    # Create a Barchart showing the count of messages by Genre
    # Create a Scatterplot showing the count of individual words from a message by Average, Sums, and Standard Deviations
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Scatter(
                    x=message_sums,
                    y=message_avgs,
                    mode='markers',
                    text=category_names,
                    marker={'size': message_stds*100}
                )
            ],

            'layout': {
                'title': 'Distribution of Message Sums and Averages',
                'yaxis': {
                    'title': "Average Occurence of Message Containing Word"
                },
                'xaxis': {
                    'title': "Sum of total Message Containing Word"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    """web page that handles user query and displays model results
    Args:
        None
    Returns:
        None
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()