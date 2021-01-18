# Disaster Response Pipelines
Udacity's Data Science Nanodegree project. 
Here an ETL will be build to read the dataset provided, clean the data, and then store it in a SQLite database. Then a machine learning pipeline will be created that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). The model will be exported to a pickle file and in the last step, the results will be displayed in a Flask web app

## Project Components
The project will have 3 main parts:

1. ETL Pipeline (```process_data.py```):
    + Loads the ```messages``` and ```categories``` datasets
    + Merges the two datasets
    + Cleans the data
    + Stores it in a SQLite database
    
2. ML Pipeline (```train_classifier.py```):
    + Loads data from the SQLite database
    + Splits the dataset into training and test sets
    + Builds a text processing and machine learning pipeline
    + Trains and tunes a model using GridSearchCV
    + Outputs results on the test set
    + Exports the final model as a pickle file
    
3. Flask Web App
    + Modify file paths for database and model
    + Add data visualizations using Plotly in the web app
