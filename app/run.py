import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objs as go
from plotly.graph_objs import Bar, Scatter
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
df = pd.read_sql_table('disaster_msg_cats', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    ## for genre plot
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    ## for number of message categories grouped by genre
    df_cats = pd.melt(df.iloc[:, 2:], id_vars=['genre'], value_vars = df.iloc[:, 3:].columns.tolist())
    df_cats = df_cats[df_cats['value']==1].groupby(['genre','variable']).sum().reset_index().set_index('genre')
    df_direct = df_cats[df_cats.index=='direct']
    df_news = df_cats[df_cats.index=='news']
    df_social = df_cats[df_cats.index=='social']
    
                        
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
                },
                'width': 700,
                'height': 500,
                'margin': 100
            }
        },
        
        {
            'data': [
                Scatter(
                    name="direct",
                    x=df_direct['variable'],
                    y=df_direct['value']
                ),
                Scatter(
                    name="news",
                    x=df_news['variable'], 
                    y=df_news['value']
                ),
                Scatter(
                    name="social",
                    x=df_social['variable'], 
                    y=df_social['value']
                )
            ],

            'layout': {
                'title': 'The Volumn of Each Category Grouped By Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
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