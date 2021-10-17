from flask import Flask, render_template, request
from flask_cors import cross_origin
import pickle
import pandas as pd
from src.application_logger.app_logger import Logger

app = Flask(__name__, template_folder='templates')

logfile_path = 'src/LogFiles/prediction_log.txt'
logger_object = Logger()


@app.route('/', methods=['GET','POST'])
@cross_origin()
def home_page():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
@cross_origin()
def result_page():
    logfile = open(logfile_path, mode='a')
    logger_object.log(logfile, "Preparing for obtaining user input.'")
    if request.method == 'POST':
        try:
            sex = str(request.form['sex'])
            age = float(request.form['age'])
            fare = float(request.form['fare'])
            family = str(request.form['family'])
            pclass = float(request.form['pclass'])
            logger_object.log(logfile, "User input obtained successfully.")
            logfile.close()
        except Exception as e:
            logger_object.log(logfile, "Exception occured while obtaining user input. Exception :" + str(e))
            logger_object.log(logfile, "Failed to obtain user input.")
            logfile.close()
            raise Exception()

        try:
            logfile = open(logfile_path, mode='a')
            df_pred = pd.DataFrame(index=[1])

            # Input for Sex
            if sex == 'Male':
                df_pred['Sex'] = 1
            else:
                df_pred['Sex'] = 0

            # Input for Age
            df_pred['Age'] = age

            # Input for Fare
            df_pred['Fare'] = fare

            # Input for Family
            if family == "Yes":
                df_pred['Family'] = 1
            else:
                df_pred['Family'] = 0

            # Input for Pclass
            if pclass == 1:
                df_pred['Pclass_2'] = 0
                df_pred['Pclass_3'] = 0
            elif pclass == 2:
                df_pred['Pclass_2'] = 1
                df_pred['Pclass_3'] = 0
            else:
                df_pred['Pclass_2'] = 0
                df_pred['Pclass_3'] = 1

            df_pred.to_csv('src/UserInput/test.csv') # Saving user input as csv
            logger_object.log(logfile, "Successfully converted User input to 'test.csv'.")
            logfile.close()
        except Exception as e:
            logger_object.log(logfile, "Exception occured during creation of 'test.csv'. Exception :" + str(e))
            logger_object.log(logfile, "Failed to create 'test.csv'")
            logfile.close()
            raise Exception()

        try:
            logfile = open(logfile_path, mode='a') # Loading the model
            model_file = 'models/decision_tree_model.pickle'
            loaded_model = pickle.load(open(model_file, 'rb'))
            logger_object.log(logfile, "Successfully loaded the model for prediction.")
            logfile.close()
        except Exception as e:
            logger_object.log(logfile, "Exception occured while loading the model. Exception :" + str(e))
            logger_object.log(logfile, "Failed to load the model for prediction.")
            logfile.close()
            raise Exception()

        try:
            logfile = open(logfile_path, mode='a')
            prediction = loaded_model.predict(df_pred)
            logger_object.log(logfile, "Successfully predicted the output.")
            logfile.close()
            print("Prediction is :", prediction)
            return render_template("result.html", prediction=prediction[0])
        except Exception as e:
            logger_object.log(logfile, "Exception occured while predicting the output. Exception :" + str(e))
            logger_object.log(logfile, "Failed to predict the output.")
            logfile.close()
            raise Exception()
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run()