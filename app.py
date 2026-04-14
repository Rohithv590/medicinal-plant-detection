from flask import Flask, render_template, request
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
import logging

app = Flask(__name__)

# Logging (STM feature)
logging.basicConfig(level=logging.INFO)

# Fix graph issue (for TF1.x)
graph = tf.get_default_graph()

with graph.as_default():
    model = load_model("model/model.h5")

classes = ["aloe", "mint", "neem", "tulsi"]

plant_info = {

    "aloe": {
        "name": "Aloe Vera",
        "scientific": "Aloe barbadensis",
        "uses": "Skin care, burns, digestion",
        "description": "Aloe Vera is used for skin healing and cooling.",
        "benefits": "Anti-inflammatory, healing, cooling",
        "how": "Apply gel or drink juice"
    },

    "mint": {
        "name": "Mint",
        "scientific": "Mentha",
        "uses": "Cold, cough, digestion",
        "description": "Mint helps digestion and gives cooling effect.",
        "benefits": "Fresh breath, digestion",
        "how": "Use in tea or food"
    },

    "neem": {
        "name": "Neem",
        "scientific": "Azadirachta indica",
        "uses": "Skin disease, blood purification",
        "description": "Neem is antibacterial medicinal plant.",
        "benefits": "Antibacterial, antifungal",
        "how": "Use paste or neem water"
    },

    "tulsi": {
        "name": "Tulsi",
        "scientific": "Ocimum tenuiflorum",
        "uses": "Cold, cough, immunity",
        "description": "Tulsi is holy basil used in Ayurveda.",
        "benefits": "Boost immunity",
        "how": "Drink tulsi tea"
    }
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # ✅ Input validation
    if "image" not in request.files:
        return render_template("index.html", error="No file uploaded")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", error="No file selected")

    # ✅ File type validation
    allowed_extensions = ["jpg", "jpeg", "png"]
    filename = file.filename.lower()

    if not any(filename.endswith(ext) for ext in allowed_extensions):
        return render_template("index.html", error="Invalid file type")

    try:
        # Read image
        img = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        # Preprocess
        img = cv2.resize(img, (160, 160))
        img = img / 255.0
        img = np.reshape(img, (1, 160, 160, 3))

        # Prediction
        with graph.as_default():
            pred = model.predict(img)

        plant = classes[np.argmax(pred)]
        confidence = float(np.max(pred)) * 100

        # Logging
        logging.info(f"Prediction: {plant}, Confidence: {confidence:.2f}%")

        # ✅ Confidence validation
        if confidence < 50:
            return render_template("index.html", error="Low confidence prediction")

        # ✅ Safe dictionary access
        info = plant_info.get(plant, None)

        if info is None:
            return render_template("index.html", error="Plant not found")

        return render_template(
            "index.html",
            info=info,
            prediction=plant,
            confidence=round(confidence, 2)
        )

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return render_template("index.html", error="Error processing image")


if __name__ == "__main__":
    app.run(debug=True)