# Disaster_Response_Project

### Note to the reviewer:  I was unable to upload the data csv files probably due to their sizes.  Thank you for your understanding.

### Description of Files:
data/process_data.py -- cleans up the input data files and saves the data to SQL database.
models/train_classifier -- Reads the SQL database and trains a model which classifies text message to categories.
app/run.py -- Runs the web app which displays master.html (displays plots) and a query box for message input.  The model classifies the input and displays matched categories in go.html.
app/templates/master.html, go/html -- explained above.

### Instructions to run the scripts:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
