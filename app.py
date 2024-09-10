import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import altair as alt
import os
import requests

process_image_pix = 162

def download_new_version_of_model():
    # URL to download from
    url = "https://drive.usercontent.google.com/download?id=10SF_y4Gb-7xSDTiBAaWgp4KtRdQd7Lg4&export=download&authuser=0&confirm=t&uuid=ff412ef3-252f-44ca-8b26-8ed400be617d&at=AO7h07fQT85gkQj02wvMRcb7ztVq:1725860542562"
    
    # Define folder and file paths
    folder_name = "fruit-360_model"
    file_name = "fruit-360_model.npy"

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Full path to save the file
    file_path = os.path.join(folder_name, file_name)

    # Streamlit progress bar and status text
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Download the file with streaming
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))  # Total size in bytes
    block_size = 1024  # 1 Kilobyte
    downloaded_size = 0  # Track the downloaded size

    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded_size += len(data)
                # Update progress bar and status text
                progress = int(downloaded_size / total_size * 100)
                progress_bar.progress(progress / 100)
                status_text.text(f"Downloaded {round(downloaded_size/1024/1024,2)} MB of {round(total_size/1024/1024,2)} MB ({progress}%)")

        # Clear progress bar and status text after download is complete
        progress_bar.empty()
        status_text.empty()
        st.success(f"File downloaded successfully and saved to {file_path}")
    else:
        progress_bar.empty()  # Ensure progress bar disappears on failure too
        st.error("Failed to download the file")


# Title and Description
st.title("Food Image Classification App")
st.write("Upload an image to get the food classification")

# Sidebar for user input
st.sidebar.title("User Input")
uploaded_file = st.sidebar.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

# Class index to food name mapping
class_names = ['Guava 1', 'Physalis 1', 'Pineapple Mini 1', 'Potato White 1', 'Zucchini 1', 'Grapefruit White 1', 'Tangelo 1', 'Lemon 1', 'Ginger Root 1', 'Tamarillo 1', 'Melon Piel de Sapo 1', 'Tomato 4', 'Strawberry 1', 'Peach 2', 'Grape White 3', 'Pomelo Sweetie 1', 'Avocado ripe 1', 'Pepino 1', 'Carambula 1', 'Avocado 1', 'Cherry 1', 'Pear Kaiser 1', 'Quince 1', 'Apricot 1', 'Peach 1', 'Mulberry 1', 'Cactus fruit 1', 'Tomato Maroon 1', 'Apple hit 1', 'Pitahaya Red 1', 'Pear Red 1', 'Apple Braeburn 1', 'Cucumber 1', 'Banana Red 1', 'Onion Red 1', 'Apple Red Yellow 2', 'Physalis with Husk 1', 'Eggplant long 1', 'Apple 6', 'Banana 1', 'Redcurrant 1', 'Plum 1', 'Nectarine 1', 'Orange 1', 'Potato Sweet 1', 'Grape White 2', 'Cherry Wax Yellow 1', 'Cucumber Ripe 2', 'Lemon Meyer 1', 'Cherry Rainier 1', 'Cabbage white 1', 'Blueberry 1', 'Hazelnut 1', 'Dates 1', 'Potato Red 1', 'Mango 1', 'Corn Husk 1', 'Onion White 1', 'Lychee 1', 'Mangostan 1', 'Rambutan 1', 'Cherry 2', 'Pepper Yellow 1', 'Cantaloupe 2', 'Grape White 4', 'Tomato Heart 1', 'Cherry Wax Black 1', 'Apple Red 1', 'Apple Red 2', 'Granadilla 1', 'Cucumber 3', 'Apple Red Delicious 1', 'Kaki 1', 'Clementine 1', 'Tomato Cherry Red 1', 'Pear 1', 'Passion Fruit 1', 'Chestnut 1', 'Nut Pecan 1', 'Pear Forelle 1', 'Apple Golden 1', 'Beetroot 1', 'Pepper Orange 1', 'Mango Red 1', 'Potato Red Washed 1', 'Pear 3', 'Tomato 1', 'Carrot 1', 'Limes 1', 'Kiwi 1', 'Peach Flat 1', 'Apple Pink Lady 1', 'Mandarine 1', 'Cauliflower 1', 'Zucchini dark 1', 'Cucumber Ripe 1', 'Papaya 1', 'Onion Red Peeled 1', 'Watermelon 1', 'Kohlrabi 1', 'Grape Pink 1', 'Nut Forest 1', 'Pear 2', 'Fig 1', 'Tomato not Ripened 1', 'Pineapple 1', 'Tomato Yellow 1', 'Apple Golden 2', 'Pear Monster 1', 'Grape Blue 1', 'Grapefruit Pink 1', 'Strawberry Wedge 1', 'Salak 1', 'Kumquats 1', 'Huckleberry 1', 'Pomegranate 1', 'Cantaloupe 1', 'Pepper Red 1', 'Banana Lady Finger 1', 'Corn 1', 'Pear Williams 1', 'Plum 2', 'Nectarine Flat 1', 'Raspberry 1', 'Apple Golden 3', 'Maracuja 1', 'Walnut 1', 'Plum 3', 'Apple Crimson Snow 1', 'Apple Granny Smith 1', 'Pear Abate 1', 'Tomato 2', 'Apple Red Yellow 1', 'Pear Stone 1', 'Cocos 1', 'Apple Red 3', 'Pepper Green 1', 'Grape White 1', 'Cherry Wax Red 1', 'Eggplant 1', 'Tomato 3']

# Function to perform convolution
def conv2d(image, kernel, bias):
    output_height = image.shape[0] - kernel.shape[0] + 1
    output_width = image.shape[1] - kernel.shape[1] + 1
    output_channels = kernel.shape[3]
    
    output = np.zeros((output_height, output_width, output_channels))
    
    for k in range(output_channels):
        for i in range(output_height):
            for j in range(output_width):
                image_patch = image[i:i + kernel.shape[0], j:j + kernel.shape[1], :]
                kernel_slice = kernel[:, :, :, k]
                if image_patch.shape[2] != kernel_slice.shape[2]:
                    raise ValueError(f"Mismatch in input channels: image has {image_patch.shape[2]}, kernel has {kernel_slice.shape[2]}")
                output[i, j, k] = np.sum(image_patch * kernel_slice) + bias[k]
    
    return output

# Function to perform max pooling
def max_pool(image, pool_size):
    output = np.zeros((image.shape[0] // pool_size[0], image.shape[1] // pool_size[1], image.shape[2]))
    for c in range(image.shape[2]):
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j, c] = np.max(image[i * pool_size[0]:(i + 1) * pool_size[0], j * pool_size[1]:(j + 1) * pool_size[1], c])
    return output

# Function to apply ReLU activation
def relu(x):
    return np.maximum(0, x)

# Function to apply softmax activation
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def add_alpha_channel(image_data):
    # Create an alpha channel with the same height and width as the image_data
    height, width, _ = image_data.shape
    alpha_channel = np.ones((height, width, 1))
    image_data_with_alpha = np.concatenate((image_data, alpha_channel), axis=-1)
    return image_data_with_alpha

def crop_center(image, new_size):
    # Crop the image to a square of size new_size, centered.
    width, height = image.size
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def flatten(pool2):
    # Ensure you are flattening the output from the last pooling layer correctly
    return pool2.reshape(-1)

if uploaded_file is not None:
    download_new_version_of_model()
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Process The Uploaded Image to fit for the model
    image = image.convert('RGB')

    min_dim = min(image.size)
    cropped_image = crop_center(image, min_dim)

    resized_image = cropped_image.resize((process_image_pix, process_image_pix))
    image_data = np.array(resized_image) / 255.0
    image_data = add_alpha_channel(image_data)
    image_data = image_data.reshape(1, process_image_pix, process_image_pix, 4)

    st.image(resized_image, caption='Edited Image (100x100)', use_column_width=True)

    # Load the pre-trained model weights
    def load_model():
        try:
            weights_dict = np.load('./fruit-360_model/fruit-360_model.npy', allow_pickle=True).item()
            return weights_dict
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    model_weights = load_model()

    if model_weights:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Perform the forward pass with progress updates
        def predict(image_data, model_weights):
            # First convolutional layer
            status_text.text("Processing: First Convolutional Layer...")
            conv1 = conv2d(image_data[0], model_weights['conv1_kernel'], model_weights['conv1_bias'])
            conv1 = relu(conv1)
            pool1 = max_pool(conv1, (2, 2))
            progress_bar.progress(25)

            # Second convolutional layer
            status_text.text("Processing: Second Convolutional Layer...")
            conv2 = conv2d(pool1, model_weights['conv2_kernel'], model_weights['conv2_bias'])
            conv2 = relu(conv2)
            pool2 = max_pool(conv2, (2, 2))
            progress_bar.progress(50)

            # Third convolutional layer
            status_text.text("Processing: Third Convolutional Layer...")
            conv3 = conv2d(pool2, model_weights['conv3_kernel'], model_weights['conv3_bias'])
            conv3 = relu(conv3)
            pool3 = max_pool(conv3, (2, 2))
            progress_bar.progress(65)

            # Fourth convolutional layer
            status_text.text("Processing: Fourth Convolutional Layer...")
            conv4 = conv2d(pool3, model_weights['conv4_kernel'], model_weights['conv4_bias'])
            conv4 = relu(conv4)
            pool4 = max_pool(conv4, (2, 2))
            progress_bar.progress(75)

            print("Shape of pool4 before flattening:", pool4.shape)
            # Flatten the output from the last pooling layer
            status_text.text("Processing: Flattening Layer...")
            flat = flatten(pool4)
            progress_bar.progress(80)
            print("Shape of pool4 after flattening:", flat.shape)

            # Fully connected layer 1 (fcl1)
            status_text.text("Processing: Fully Connected Layer 1...")
            fcl1 = np.dot(flat, model_weights['fcl1_kernel']) + model_weights['fcl1_bias']
            fcl1 = relu(fcl1)
            progress_bar.progress(85)

            # Fully connected layer 2 (fcl2)
            status_text.text("Processing: Fully Connected Layer 2...")
            fcl2 = np.dot(fcl1, model_weights['fcl2_kernel']) + model_weights['fcl2_bias']
            fcl2 = relu(fcl2)
            progress_bar.progress(90)

            # Output layer (predictions)
            status_text.text("Processing: Output Layer...")
            predictions = np.dot(fcl2, model_weights['predictions_kernel']) + model_weights['predictions_bias']
            output = softmax(predictions)
            progress_bar.progress(100)

            return output

        predictions = predict(image_data, model_weights)
        predicted_class = np.argmax(predictions)
        predicted_food = class_names[predicted_class]

        st.write(f"Predicted Class Index: {predicted_class}")
        st.write(f"Predicted Food: {predicted_food}")

        progress_bar.empty()
        status_text.empty()
    else:
        st.error("Failed to load model weights.")

slider_value = st.slider("Select a value", 0, 100)
text_input = st.text_input("Enter some text")

# Progress and status updates with delay
progress_bar = st.progress(0)
for i in range(100):
    time.sleep(0.02)
    progress_bar.progress(i + 1)

data = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
chart = alt.Chart(data).mark_line().encode(x='a', y='b')
st.altair_chart(chart)
