from django.shortcuts import render

import pandas as pd
import joblib


# Load trained model
MODEL_PATH = r"d:\Downloads\laptop_price_prediction_model.pkl"

model = joblib.load(MODEL_PATH)


def index(request):

    prediction = None

    if request.method == "POST":

        ram = float(request.POST["ram"])

        screen = float(request.POST["screen"])

        weight = float(request.POST["weight"])

        battery = float(request.POST["battery"])


        input_data = pd.DataFrame({

            "RAM_GB": [ram],

            "Screen_Size_Inches": [screen],

            "Weight_KG": [weight],

            "Battery_Hours": [battery]

        })


        prediction = model.predict(input_data)[0]

        prediction = round(prediction)


    return render(

        request,

        "index.html",

        {

            "prediction": prediction

        }

    )
