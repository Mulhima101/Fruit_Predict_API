from fastapi import FastAPI
import tensorflow as tf
import numpy as np

app = FastAPI()

# Load the pre-trained model
# model = tf.keras.models.load_model('mnist_model.h5')

@app.get("/")
def home():
    # Load the MNIST dataset for prediction
    # (_, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    # test_images = test_images / 255.0

    # Make predictions
    # predictions = model.predict(test_images)
    # prediction_for_first_image = np.argmax(predictions[0])

    return {"prediction_for_first_image": 6}
