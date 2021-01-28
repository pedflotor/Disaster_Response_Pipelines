import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download(['stopwords'])
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/disaster_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Most frequent categories
    df_categories = df.drop(['id'], axis=1)._get_numeric_data()
    top_categories_pcts = df_categories.sum().sort_values(ascending=False).head(15)
    top_categories_names = list(top_categories_pcts.index)

    # Most frequent words
    social_media_messages = ' '.join(df[df['buildings'].astype(str) == '1']['message'])
    print('Tokenize messages...15% completed')
    tokens_partial = [w for w in word_tokenize(social_media_messages.lower()) if w.isalpha()]
    print('Removing stopwords...30% completed')
    cachedStopWords = stopwords.words("english")
    text_partial = [word for word in tokens_partial if word not in cachedStopWords]
    print('Removing most common words...45% completed')
    social_media_wrd_counter = Counter(text_partial).most_common()
    social_media_wrd_cnt = [i[1] for i in social_media_wrd_counter]
    social_media_wrd_pct = [i / sum(social_media_wrd_cnt) * 100 for i in social_media_wrd_cnt]
    social_media_wrds = [i[0] for i in social_media_wrd_counter]

    # Most frequent words
    messages = ' '.join(df['message'])
    print('Tokenize messages...60% completed')
    tokens = [w for w in word_tokenize(messages.lower()) if w.isalpha()]
    print('Removing stopwords...75% completed')
    cachedStopWords = stopwords.words("english")
    text = [word for word in tokens if word not in cachedStopWords]
    print('Removing most common words...100% completed')
    wrd_counter = Counter(text).most_common()
    wrd_cnt = [i[1] for i in wrd_counter]
    wrd_pct = [i / sum(wrd_cnt) * 100 for i in wrd_cnt]
    wrds = [i[0] for i in wrd_counter]

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
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
                Bar(
                    x=top_categories_names,
                    y=top_categories_pcts
                )
            ],

            'layout': {
                'title': 'Top 15 Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=social_media_wrds[:25],
                    y=social_media_wrd_pct[:25]
                )
            ],

            'layout': {
                'title': "Top 25 Keywords in Buildings category",
                'xaxis': {'tickangle': 60
                          },
                'yaxis': {
                    'title': "% Total Social Media Messages"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=wrds[:25],
                    y=wrd_pct[:25]
                )
            ],

            'layout': {
                'title': "Top 25 Keywords in the messages",
                'xaxis': {'tickangle': 60
                          },
                'yaxis': {
                    'title': "% Total Social Media Messages"
                }
            }
        }

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
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