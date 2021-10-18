from flask import Flask, render_template, request
from flask_cors import cross_origin
import pickle
import pandas as pd

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET','POST'])
@cross_origin()
def home_page():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
@cross_origin()
def result_page():

    if request.method == 'POST':
        try:
            sex = str(request.form['sex'])
            age = float(request.form['age'])
            fare = float(request.form['fare'])
            family = str(request.form['family'])
            pclass = float(request.form['pclass'])

        except Exception as e:

            raise Exception()

        try:

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

        except Exception as e:

            raise Exception()

        try:

            model_file = 'models/decision_tree_model.pickle'
            loaded_model = pickle.load(open(model_file, 'rb'))

        except Exception as e:

            raise Exception()

        try:

            prediction = loaded_model.predict(df_pred)

            print("Prediction is :", prediction)
            return render_template("result.html", prediction=prediction[0])
        except Exception as e:
            raise Exception()
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run()