import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("C://Users//aisup//Downloads//Efficientnet_model.hdf5")

# Map numerical class labels to text labels
class_labels = {
    0: "Bread",
    1: "Dairy product",
    2: "Dessert",
    3: "Egg",
    4: "Fried food",
    5: "Meat",
    6: "Noodles/Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable/Fruit"
}

# Custom CSS styling
st.markdown(
    """
    <style>
    .app-title {
        font-size: 36px;
        color: #f26522;
        text-align: center;
        margin-bottom: 30px;
    }
    .upload-box {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .button {
        background-color: #f26522;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }
    .button:hover {
        background-color: #d44d1e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app header with enhanced styling
st.markdown('<h1 class="app-title">📷 Image Classification App</h1>', unsafe_allow_html=True)
st.markdown(
    "Upload an image and let the app predict the class label. "
    "Choose from formats: JPG, JPEG, PNG."
)

# Upload image through Streamlit UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image")

if uploaded_file is not None:
    # Classification button
    if st.button("Predict", key="predict_button"):
        # Read and preprocess image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        # Make predictions
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions)  # Keep numerical class label

        # Display prediction results above the image
        st.markdown(
            f'<h2 style="color: #f26522;">Predicted Class: {class_labels[predicted_class]}</h2>',
            unsafe_allow_html=True
        )
        st.image(image, caption='Predicted Image', use_column_width=True)
