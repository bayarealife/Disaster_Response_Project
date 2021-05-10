# Disaster_Response_Project

### Note to the reviewer:  I was unable to upload the data csv files probably due to their sizes.  Thank you for your understanding.

### Motivations
Build a machine learning model which classifies disaster messages sourced by FigureEight into appropriate categories so the message gets rounted to the correct responders more quickly and efficiently.
Also set up a web page which a user can enter a message and look up the matched categories.  The web page also displays the analytics of the data used to train the model.

### Description of Files:
data/process_data.py -- cleans up the input data files and saves the data to SQL database.
models/train_classifier -- Reads the SQL database and trains a model which classifies text message to categories.
app/run.py -- Runs the web app which displays master.html (displays plots) and a query box for message input.  The model classifies the input and displays matched categories in go.html.
app/templates/master.html, go/html -- explained above.

### Installation
Copy files to your desired folder maintaining the following file structure:
|-- app folder
    |-- templates
        |-- go.html
        |-- master.html
    |--run.py
|-- models folder
    |-- train_classifier.py
|-- data folder
    |-- process_data.py
    |-- disaster_categories.csv (not present in this repository due to upload issue)
    |-- disaster_messages.csv (not present in this repository due to upload issue)

### Instructions to run the scripts:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

![image](https://user-images.githubusercontent.com/18743274/117695977-aad04d00-b175-11eb-9b17-d85bec33ab50.png)


### Acknowledgement
All the fellow Udacity Data Science Program students for sharing their knowledge and experience and the wonderful instructors!!


