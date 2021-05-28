"""module for flask api"""
import os

import numpy as np
import tensorflow.keras as tfk
from flask import Flask, redirect, render_template, request, session, url_for
from flask_uploads import IMAGES, UploadSet, configure_uploads

from flask_session import Session
from src.model import Mnist

app = Flask(__name__)
SESSION_TYPE = "filesystem"
app.config.from_object(__name__)
Session(app)

photos = UploadSet("photos", IMAGES)

# path for saving uploaded images
app.config["UPLOADED_PHOTOS_DEST"] = "./static/img"
configure_uploads(app, photos)

# Home page
@app.route("/", methods=["GET", "POST"])
@app.route("/home", methods=["GET", "POST"])
def home():
    """Home page for the WEB-API

    Returns:
        str: welcome message
    """
    welcome = "Fashion MNIST API"
    return welcome


model = Mnist()
model.load_model("save_model")

# route for image inference
@app.route("/infer", methods=["GET", "POST"])
def infer():
    """Page for sigle image inference"""
    # pylint: disable = no-else-return
    if request.method == "POST":
        if request.form.get("eval"):
            return redirect(url_for("evaluate"))
        elif "photo" in request.files:
            if request.files["photo"]:

                # save the image
                filename = photos.save(request.files["photo"])
                filepath = os.path.join("./static/img", filename)

                # load the image
                img = tfk.preprocessing.image.load_img(filepath, color_mode="grayscale")
                img = tfk.preprocessing.image.img_to_array(img, dtype=float) / 255.0

                # Cleanup
                os.remove(filepath)

                # return prediction
                return model.infer(img)
            else:
                return "Please provide a valid image file"

    # web page before the POST request
    return render_template("infer.html")


@app.route("/eval", methods=["GET", "POST"])
def evaluate():
    """Page for model evaluation"""
    if request.method == "POST":
        # pylint: disable = no-else-return
        if request.form.get("execute"):
            return model.eval()
        elif request.form.get("infer"):
            return redirect(url_for("infer"))

    # web page before the POST request
    return render_template("eval.html")


@app.route("/train", methods=["GET", "POST"])
def train():
    """page for model training"""

    # pylint: disable = no-else-return
    if request.method == "POST":

        # extract variable from the request
        epochs = int(request.form.get("epochs"))
        train_batch_size = int(request.form.get("train"))
        l_rate = 1e-3 * float(request.form.get("lr"))

        # commence training
        session["history"] = model.train(
            epochs=epochs,
            train_batch_size=train_batch_size,
            l_rate=l_rate,
            cache=True,
        )

        return redirect(url_for("train_finish"))

    # web page for train inputs
    return render_template("train.html")


@app.route("/train/finish", methods=["GET", "POST"])
def train_finish():
    """training finish page"""
    history = session.get("history")
    index = np.argmax(history["val_accuracy"])
    val_accuracy = history["val_accuracy"][index]
    val_loss = history["val_loss"][index]

    # pylint: disable = no-else-return
    if request.method == "POST":
        # Redirects after training
        if request.form.get("infer"):
            return redirect(url_for("infer"))
        elif request.form.get("eval"):
            return redirect(url_for("evaluate"))

    return render_template(
        "train_finish.html",
        val_loss=f"{val_loss:.4f}",
        val_accuracy=f"{val_accuracy:.4f}",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0")
