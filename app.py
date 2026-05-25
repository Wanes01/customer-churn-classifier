from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# carica il modello, encoder e le feature categoriche
with open("model/best_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)
with open("model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
    
# route che punta alla pagina html col form
@app.route("/")
def index():
    return render_template("index.html")

# route per richiedere la predizione di churn
@app.route("/predict", methods=["POST"])
def prefict():
    data = request.get_json()
    
    # feature binarie direttamente mappabili
    num_data = {
        "tenure": int(data["tenure"]),
        "MonthlyCharges": float(data["monthly_charges"]),
        "SeniorCitizen": int(data["senior_citizen"]),
        "Partner": 1 if data["partner"] == "Yes" else 0,
        "Dependents": 1 if data["dependents"] == "Yes" else 0,
        "PaperlessBilling": 1 if data["paperless_billing"] == "Yes" else 0,
        "PhoneService": 1 if data["phone_service"] == "Yes" else 0,
        "GenderMale": 1 if data["gender"] == "Male" else 0,
    }
    num_df = pd.DataFrame([num_data])
    
    # le feature categoriche da passare all'encoder
    cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
    ]
    cat_data = {col: data[col] for col in cat_cols}
    cat_df = pd.DataFrame([cat_data])
    
    # applichiamo lo stesso encoder del notebook
    enc_array = encoder.transform(cat_df)
    enc_df = pd.DataFrame(
        enc_array,
        columns=encoder.get_feature_names_out(cat_cols)
    )
    
    # uniamo e riordiniamo le feature come si
    # aspetta il modello
    input_df = pd.concat([num_df, enc_df], axis=1)[feature_names]
    
    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return jsonify({
        "prediction":  int(prediction),
        "probability": float(probability)
    })
    

if __name__ == "__main__":
    app.run(debug=True)
