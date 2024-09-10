from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Lambda
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import io
import requests

app = FastAPI()

# # Download model function
# def download_new_version_of_model():
#     url = "https://drive.usercontent.google.com/download?id=11jx-8NS5sU9tqFVgML9VOcpMaX_2ZaWl&export=download&authuser=0&confirm=t&uuid=a8c7024d-83f1-49c3-b723-6a06189f5a1c&at=AO7h07fqdlUwCe7UJ3dZBpprFe5h:1725951295439"
#     folder_name = "fruit-360_model"
#     file_name = "fruit-360_model.keras"

#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)

#     file_path = os.path.join(folder_name, file_name)
#     response = requests.get(url, stream=True)
#     total_size = int(response.headers.get('content-length', 0))
#     block_size = 1024
#     downloaded_size = 0

#     if response.status_code == 200:
#         with open(file_path, 'wb') as f:
#             for data in response.iter_content(block_size):
#                 f.write(data)
#                 downloaded_size += len(data)
#                 progress = int(downloaded_size / total_size * 100)
#                 print(f"\rDownloaded {round(downloaded_size/1024/1024,2)} MB of {round(total_size/1024/1024,2)} MB ({progress}%)", end=" ")

# download_new_version_of_model()

# Custom Lambda function for converting to HSV and Grayscale
def convert_to_hsv_and_grayscale(x):
    hsv = tf.image.rgb_to_hsv(x)
    grayscale = tf.image.rgb_to_grayscale(x)
    return tf.concat([hsv, grayscale], axis=-1)

# Register the custom function with TensorFlow
@tf.keras.utils.register_keras_serializable()
def custom_lambda(x):
    return convert_to_hsv_and_grayscale(x)

# # Use the correct decorator for registering custom objects
# @tf.keras.utils.register_keras_serializable()
# def convert_to_hsv_and_grayscale(x):
#     hsv = tf.image.rgb_to_hsv(x)
#     gray = tf.image.rgb_to_grayscale(x)
#     rez = tf.concat([hsv, gray], axis=-1)
#     return rez

# Define output shape for Lambda layer
def custom_lambda_output_shape(input_shape):
    return input_shape[:-1] + (4,)

# Load the .keras model with the custom Lambda layer
model = load_model("./fruit-360_model/fruit-360_model.keras", custom_objects={
    'convert_to_hsv_and_grayscale': convert_to_hsv_and_grayscale,
    'Lambda': Lambda(convert_to_hsv_and_grayscale, output_shape=custom_lambda_output_shape),
    'custom_lambda': custom_lambda
})

# model = load_model("./fruit-360_model/fruit-360_model.keras")

# # Example class names (adjust based on your dataset)
class_names = ['Guava 1', 'Physalis 1', 'Pineapple Mini 1', 'Potato White 1', 'Zucchini 1', 'Grapefruit White 1', 'Tangelo 1', 'Lemon 1', 'Ginger Root 1', 'Tamarillo 1', 'Melon Piel de Sapo 1', 'Tomato 4', 'Strawberry 1', 'Peach 2', 'Grape White 3', 'Pomelo Sweetie 1', 'Avocado ripe 1', 'Pepino 1', 'Carambula 1', 'Avocado 1', 'Cherry 1', 'Pear Kaiser 1', 'Quince 1', 'Apricot 1', 'Peach 1', 'Mulberry 1', 'Cactus fruit 1', 'Tomato Maroon 1', 'Apple hit 1', 'Pitahaya Red 1', 'Pear Red 1', 'Apple Braeburn 1', 'Cucumber 1', 'Banana Red 1', 'Onion Red 1', 'Apple Red Yellow 2', 'Physalis with Husk 1', 'Eggplant long 1', 'Apple 6', 'Banana 1', 'Redcurrant 1', 'Plum 1', 'Nectarine 1', 'Orange 1', 'Potato Sweet 1', 'Grape White 2', 'Cherry Wax Yellow 1', 'Cucumber Ripe 2', 'Lemon Meyer 1', 'Cherry Rainier 1', 'Cabbage white 1', 'Blueberry 1', 'Hazelnut 1', 'Dates 1', 'Potato Red 1', 'Mango 1', 'Corn Husk 1', 'Onion White 1', 'Lychee 1', 'Mangostan 1', 'Rambutan 1', 'Cherry 2', 'Pepper Yellow 1', 'Cantaloupe 2', 'Grape White 4', 'Tomato Heart 1', 'Cherry Wax Black 1', 'Apple Red 1', 'Apple Red 2', 'Granadilla 1', 'Cucumber 3', 'Apple Red Delicious 1', 'Kaki 1', 'Clementine 1', 'Tomato Cherry Red 1', 'Pear 1', 'Passion Fruit 1', 'Chestnut 1', 'Nut Pecan 1', 'Pear Forelle 1', 'Apple Golden 1', 'Beetroot 1', 'Pepper Orange 1', 'Mango Red 1', 'Potato Red Washed 1', 'Pear 3', 'Tomato 1', 'Carrot 1', 'Limes 1', 'Kiwi 1', 'Peach Flat 1', 'Apple Pink Lady 1', 'Mandarine 1', 'Cauliflower 1', 'Zucchini dark 1', 'Cucumber Ripe 1', 'Papaya 1', 'Onion Red Peeled 1', 'Watermelon 1', 'Kohlrabi 1', 'Grape Pink 1', 'Nut Forest 1', 'Pear 2', 'Fig 1', 'Tomato not Ripened 1', 'Pineapple 1', 'Tomato Yellow 1', 'Apple Golden 2', 'Pear Monster 1', 'Grape Blue 1', 'Grapefruit Pink 1', 'Strawberry Wedge 1', 'Salak 1', 'Kumquats 1', 'Huckleberry 1', 'Pomegranate 1', 'Cantaloupe 1', 'Pepper Red 1', 'Banana Lady Finger 1', 'Corn 1', 'Pear Williams 1', 'Plum 2', 'Nectarine Flat 1', 'Raspberry 1', 'Apple Golden 3', 'Maracuja 1', 'Walnut 1', 'Plum 3', 'Apple Crimson Snow 1', 'Apple Granny Smith 1', 'Pear Abate 1', 'Tomato 2', 'Apple Red Yellow 1', 'Pear Stone 1', 'Cocos 1', 'Apple Red 3', 'Pepper Green 1', 'Grape White 1', 'Cherry Wax Red 1', 'Eggplant 1', 'Tomato 3']

# # Preprocessing function for incoming images
# def preprocess_image(image_data: Image.Image, target_size: tuple):
#     image_data = image_data.resize(target_size)
#     img_array = image.img_to_array(image_data)
#     img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to fit model input
#     img_array /= 255.0  # Normalize the image data
#     return img_array

def crop_center(image):
    # Crop the image to a square of size new_size, centered.
    new_size = min(image.size)
    width, height = image.size
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def preprocess_image(img_data, target_size=(100, 100)):
    img = crop_center(img_data)
    img = img.resize(target_size)  # Resize to match model input
    img_array = tf.keras.utils.array_to_img(img)  # Convert image to array
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255 # Normalize image
    return img_array

# # Function to load the model, predict, and output the class label
# def load_and_predict_image(image_data, model_out_dir, labels):
#     # Load model weights
#     model = keras.models.load_model(model_out_dir + "/fruit-360_model.keras",
#                                     custom_objects={"convert_to_hsv_and_grayscale": convert_to_hsv_and_grayscale})

#     # Normalize the image array
#     img_array = preprocess_image(image_data)
#     # img_array = img_array / 255.0  # Normalize pixel values to [0, 1]

#     # Perform prediction
#     predictions = model.predict(img_array)
    
#     # Get the predicted class index with highest probability
#     score= tf.nn.softmax(predictions)

#     # Check if predicted_class_index is in the range of labels
#     if score >= len(labels):
#         return {"error": "Predicted class index is out of range."}
    
#     predicted_label = labels[np.argmax(score)]
#     print("lable: ", predicted_label)
    
#     return predicted_label

@app.get("/")
async def predict():
    return "Use Postment to upload a image and see the results."

@app.post("/")
async def predict(file: UploadFile = File(...)):
    try:
        # # Read and process the uploaded image
        image_data = await file.read()
        image_data = Image.open(io.BytesIO(image_data)).convert('RGB')

        # result = load_and_predict_image(image_data, "./fruit-360_model", class_names)
        
        # Resize and preprocess the image
        processed_image = preprocess_image(image_data, target_size=(100, 100))

        # Make predictions using the loaded model
        predictions = model.predict(processed_image)

        # Debugging: Print the prediction results and their shape
        print(f"Predictions: {predictions}")
        print(f"Prediction shape: {predictions.shape}")
        
        predicted_class = np.argmax(predictions, axis=1)[0]
        
        # Debugging: Print the predicted class index
        print(f"Predicted class index: {predicted_class}")

        # Make sure the predicted class index is within bounds of class_names
        if predicted_class < len(class_names):
            predicted_class_name = class_names[predicted_class]
        else:
            return {"error": "Predicted class index is out of range."}

        # Return the prediction result
        return {
            # "predicted_class_index": int(confidence_scores),
            "predicted_class_name": predicted_class_name,
            # "confidence_scores": predictions.tolist()
        }

    except Exception as e:
        return {"error": str(e)}
