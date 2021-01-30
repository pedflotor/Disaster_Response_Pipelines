import json
import plotly
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
nltk.download(['stopwords'])


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
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("models/disaster_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Most repeated categories
    df_cat = df.drop(['id'], axis=1)._get_numeric_data()
    top_cat_pcts = df_cat.sum().sort_values(ascending=False).head(25)
    top_cat_names = list(top_cat_pcts.index)

    def filter_words(text):
        """
        Function to case normalize, lemmatize, remove stopwords and tokenize text using nltk
        It also calculates the percentage that each word represents in the total, taken into account all the words found

        Parameters
        ----------
        text:object
          text data with the raw messages that will be processed

     Returns
        -------
        word_percentage:list
          list of the percentages that each word have between all words found
        words: list
          list of words ordered from the most repeated one
      """

        tokens = [w for w in word_tokenize(text.lower()) if w.isalpha()]
        print('Removing stopwords...30% completed')
        cachedStopWords = stopwords.words("english")
        text_processed = [word for word in tokens if word not in cachedStopWords]
        print('Removing most common words...60% completed')
        word_counter = Counter(text_processed).most_common()
        print('Removing most common words...100% completed')
        word_count = [i[1] for i in word_counter]
        word_percentage = [i / sum(word_count) * 100 for i in word_count]
        words = [i[0] for i in word_counter]

        return word_percentage, words

    print('Number of tasks to be completed: 5')
    print('Estimated time of completion: 40 seconds')

    # Most frequent words when buildings is set to 1
    print('Task 1')
    b_messages = ' '.join(df[df['buildings'].astype(str) == '1']['message'])
    b_pct, b_wrd = filter_words(b_messages)

    # Most frequent words when buildings is set to 0
    print('Task 2')
    b_n_messages = ' '.join(df[df['buildings'].astype(str) == '0']['message'])
    b_n_pct, b_n_wrd = filter_words(b_n_messages)

    # Most frequent words when related is set to 1
    print('Task 3')
    r_messages = ' '.join(df[df['related'].astype(str) == '1']['message'])
    r_pct, r_wrd = filter_words(r_messages)

    # Most frequent words when related is set to 0
    print('Task 4')
    r_n_messages = ' '.join(df[df['related'].astype(str) == '0']['message'])
    r_n_pct, r_n_wrd = filter_words(r_n_messages)

    # Most frequent words in all the messages
    print('Task 5')
    messages = ' '.join(df['message'])
    wrd_pct, wrds = filter_words(messages)

    # create visuals
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
                    x=top_cat_names,
                    y=top_cat_pcts
                )
            ],

            'layout': {
                'title': 'Top 25 Message Categories',
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
                    x=wrds[:25],
                    y=wrd_pct[:25]
                )
            ],

            'layout': {
                'title': "Top 25 Keywords in the messages",
                'xaxis': {'tickangle': 60
                          },
                'yaxis': {
                    'title': "% Total Messages"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=b_wrd[:8],
                    y=b_pct[:8]
                )
            ],

            'layout': {
                'title': "Top 8 Keywords in Buildings category",
                'xaxis': {'tickangle': 60
                          },
                'yaxis': {
                    'title': "% Total Building Messages"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=b_n_wrd[:8],
                    y=b_n_pct[:8]
                )
            ],

            'layout': {
                'title': "Top 8 Keywords not in building category",
                'xaxis': {'tickangle': 60
                          },
                'yaxis': {
                    'title': "% Total not Building Messages"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=r_wrd[:8],
                    y=r_pct[:8]
                )
            ],

            'layout': {
                'title': "Top 8 Keywords in Related category",
                'xaxis': {'tickangle': 60
                          },
                'yaxis': {
                    'title': "% Total Related Messages"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=r_n_wrd[:8],
                    y=r_n_pct[:8]
                )
            ],

            'layout': {
                'title': "Top 8 Keywords not in Related category",
                'xaxis': {'tickangle': 60
                          },
                'yaxis': {
                    'title': "% Total not Related Messages"
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