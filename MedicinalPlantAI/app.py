from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

graph = tf.get_default_graph()

model = load_model("model/model.h5")

classes = ["aloe", "mint", "neem", "tulsi"]

plant_info = {

    "aloe": {
        "name": "Aloe Vera",
        "scientific": "Aloe barbadensis",
        "uses": "Skin care, burns, digestion",
        "description": "Aloe Vera is a medicinal plant used for skin healing and cooling.",
        "benefits": "Anti-inflammatory, healing, cooling",
        "how": "Apply gel on skin or drink juice"
    },

    "mint": {
        "name": "Mint",
        "scientific": "Mentha",
        "uses": "Cold, cough, digestion",
        "description": "Mint helps in digestion and gives cooling effect.",
        "benefits": "Fresh breath, digestion, cooling",
        "how": "Use in tea or food"
    },

    "neem": {
        "name": "Neem",
        "scientific": "Azadirachta indica",
        "uses": "Skin disease, blood purification",
        "description": "Neem is strong antibacterial medicinal plant.",
        "benefits": "Antibacterial, antifungal",
        "how": "Use leaves paste or neem water"
    },

    "tulsi": {
        "name": "Tulsi",
        "scientific": "Ocimum tenuiflorum",
        "uses": "Cold, cough, immunity",
        "description": "Tulsi is holy basil used in Ayurveda.",
        "benefits": "Boost immunity, reduce fever",
        "how": "Drink tulsi tea"
    }

}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    img = cv2.resize(img, (160, 160))

    img = img / 255.0

    img = np.reshape(img, (1, 160, 160, 3))

    with graph.as_default():
        pred = model.predict(img)

    plant = classes[np.argmax(pred)]

    info = plant_info[plant]

    return render_template(
        "index.html",
        info=info
    )


if __name__ == "__main__":
    app.run(debug=True)