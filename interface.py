import joblib
import PySimpleGUI as sg
import pandas as pd

# === Load model and expected features ===
model = joblib.load("house_trained_clean_model.pkl")
expected_features = joblib.load('clean_model_features.pkl')  # list of columns used during training

sg.theme('Tan')

layout = [
    [sg.Text("Enter house data for prediction")],
    [sg.Text("Rooms", size=(15,1)), sg.Input(key="room")],
    [sg.Text("Distance", size=(15,1)), sg.Input(key="distance")],
    [sg.Text("Bedroom", size=(15,1)), sg.Input(key="bedroom")],
    [sg.Text("Bathroom", size=(15,1)), sg.Input(key="bathroom")],
    [sg.Text("Car", size=(15,1)), sg.Input(key="car")],
    [sg.Text("Landsize", size=(15,1)), sg.Input(key="landsize")],
    [sg.Text("BuildingArea", size=(15,1)), sg.Input(key="buildingarea")],
    [sg.Text("YearBuilt", size=(15,1)), sg.Input(key="yearbuilt")],
    [sg.Text("CouncilArea", size=(15,1)), sg.Input(key="councilarea")],
    [sg.Button("Predict", key='button')],
    [sg.Text("Prediction:", size=(15,1)), sg.Text("", key="output")]
]

window = sg.Window("HOUSE PRICE PREDICTOR", layout, element_justification="center")

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    elif event == "button":
        try:
            # === Collect and convert inputs ===
            input_data = {
                "Suburb": float(values["suburb"]),
                "Rooms": float(values["room"]),
                "Type": float(values["type"]),
                "Distance": float(values["distance"]),
                "Bedroom": float(values["bedroom"]),
                "Bathroom": float(values["car"]),
                "Car": float(values["car"]),
                "Landsize": float(values["landsize"]),
                "BuildingArea": float(values["buildingarea"]),
                "Yearbuilt": float(values["yearbuilt"]),
                "CouncilArea": float(values["councilarea"])
            }

            # === Format input for model ===
            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=expected_features)
            input_df = input_df.fillna(0)

            # === Predict and show result ===
            prediction = model.predict(input_df)
            window["output"].update(f"{prediction[0]:.2f}")

        except Exception as e:
            sg.popup_error("Error:", str(e))

window.close()
