from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("clean_model.h5")

classes = ["Barren", "Urban", "Vegetation", "Water"]

def preprocess(img):
    img = img.resize((128,128))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def home():
    return "LandLens API Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file).convert("RGB")

    processed = preprocess(img)
    pred = model.predict(processed)

    result = classes[np.argmax(pred)]

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)