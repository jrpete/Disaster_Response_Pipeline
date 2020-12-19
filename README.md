# README

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Motivation
The Jupyter Notebook associated with this README uses messages sent during a disaster event that were obtained from [Figure Eight](https://appen.com/).
The Notebook transforms the data from csv files into a cleaned output that is loaded into a SQL Lite database, then feed this data into a MultiOutput Classifier Model to train the messages against a set of categories used as labels

## Libraries
See the requirements.txt for a listed of the packages used in this project


## Files
There are 3 primary files in this project:
	app/run.py: Creates a Flask app that loads the data from the database created by process_data.py and model created by train_classifier.py then displays it on a webpage using Plotly to visaulize the data. 
	
	data/process_data.py: Loads, Cleans, and Outputs the disaster_messages.csv and disaster_categories.csv files into a database.
	
	models/train_classifier.py: Loads, Tokenizes, Models, Evaluates, and Saves data from the database created by process_data.py


## Acknowledgements

This Notebook was compiled by myself for the purposes of meeting the requirements of the 'Disaster Response Pipelines' Project 
as part of Udacity's Data Science Nanodegree Program. 