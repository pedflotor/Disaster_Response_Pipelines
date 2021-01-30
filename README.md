# Disaster Response Pipelines
Udacity's Data Science Nanodegree project. 
Here an ETL will be built to read the dataset provided, clean the data, and store it in a SQLite database. Then a 
machine learning pipeline will be created that uses NLTK as well as scikit-learn's Pipeline and GridSearchCV to 
output a final model that uses the message column to predict classifications for 36 categories (multi-output 
classification). The model will be exported to a pickle file and in the last step, the results will be displayed 
in a Flask web app

## Project Components
The project has 3 main parts:

1. ETL Pipeline [(```process_data.py```)](data/process_data.py):
    + Loads the ```messages``` and ```categories``` datasets
    + Merges the two datasets
    + Cleans the data
    + Stores it in a SQLite database
    
2. ML Pipeline [(```train_classifier.py```)](models/train_classifier.py):
    + Loads data from the SQLite database
    + Splits the dataset into training and test sets
    + Builds a text processing and machine learning pipeline
    + Trains and tunes a model using GridSearchCV
    + Outputs results on the test set
    + Exports the final model as a pickle file
    
3. Flask Web App [(```run```)](app/run.py)
    + Web application where new messages can be input and get classification results in 
    different categories
    + Includes data visualizations using Plotly

## Files in the repository
* /app/templates/go.html: web page that handles user query and displays model results
* /app/templates/master.html: index webpage displays visuals and receives user input text for model
* /app/run.py: flask application to run the web
* /data/disaster_categories.csv: categories dataset
* /data/disaster_messages.csv: messages dataset
* /data/DisasterResponse.db: SQLite database containing the merged dataset
* /data/process_data.csv: data cleaning ETL pipeline that merges the two dataset aforementioned
* /models/classifier.pkl: classifier model as a pickle file
* /models/train_classifier.py: machine learning pipeline that builds the classifier model aforementioned
* /README.md: description of the project

## Instructions
1. Run the following commands in the project's root directory to set up your database and model:
    - To run the ETL pipeline that cleans the data and stores it in a SQLite database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline that trains the classifier and saves the model as a pickle file:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app:
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

## Website
![Dataset_insights_1](https://github.com/pedflotor/Disaster_Response_Pipelines/blob/main/pics/Dataset_insights_1.png)
![Dataset_insights_2](https://github.com/pedflotor/Disaster_Response_Pipelines/blob/main/pics/Dataset_insights_2.png)
![Message_Categorization_1](https://github.com/pedflotor/Disaster_Response_Pipelines/blob/main/pics/Message_Categorization_1.png)
![Message_Categorization_2](https://github.com/pedflotor/Disaster_Response_Pipelines/blob/main/pics/Message_Categorization_2.png)