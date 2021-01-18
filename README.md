# Disaster_Response_Pipelines
Udacity's Data Science Nanodegree project. 
Here an ETL will be build to read the dataset provided, clean the data, and then store it in a SQLite database. Then a machine learning pipeline will be created that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). The model will be exported to a pickle file and in the last step, the results will be displayed in a Flask web app

##Project Components
The project is compounded by 3 components:

1. ETL Pipeline (process_data.py):
*Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database
2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file
3. Flask Web App
We are providing much of the flask web app for you, but feel free to add extra features depending on your knowledge of flask, html, css and javascript. For this part, you'll need to:

Modify file paths for database and model as needed
Add data visualizations using Plotly in the web app. One example is provided for you
