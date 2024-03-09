import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('modelmobilenet.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size of the model
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image = np.array(image)
    # Normalize the pixel values to be between 0 and 1
    image = image / 255.0
    # Expand the dimensions of the image to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    return image

# Create a Streamlit web interface
st.title('Malaria Disease Detection')
st.write('Upload an image of a cell to check for malaria infection:')
uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Preprocess the uploaded image
    image = Image.open(uploaded_image)
    image_array = preprocess_image(image)
    
    # Use the model to make a prediction
    prediction = model.predict(image_array)
    
    # Display the prediction result
    if prediction[0][0] <= 0.5:
        st.write('The cell is uninfected.')
    else:
        st.write('The cell is infected with malaria.')
