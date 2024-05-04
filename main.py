import streamlit as st
import cv2 as cv
import numpy as np
import keras
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="🌱 Plant Disease Detection 🌿",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.markdown("<h1 style='text-align: center; color: #38B000;'>🌱 Plant Disease Detection 🌿</h1>", unsafe_allow_html=True)


def user_guide():
    # Main content
    st.markdown("""
    The **leaf disease detection model** is built using deep learning techniques, and it **utilizes transfer learning** to leverage the pre-trained knowledge of a base model. The model has been **trained on a dataset containing images of 33 different types of leaf diseases**. For more information about the architecture, dataset, and training process, please refer to the **code and documentation** provided.
    """)

    # Dataset use
    st.markdown(f"**Data Set use:** https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset", unsafe_allow_html=True)
    # Additional content
    st.markdown("""
    The goal of this project is to identify the disease and provide information about the image.

    ## Usage

    To use the model for leaf disease detection, follow these steps:
    1. Set up the environment
    
    ```
    python -m venv venv
    ```
    
    2. Make sure you have a Python environment set up with the necessary libraries installed. You can use the provided requirements.txt file to set up the required dependencies.
    
    ```
    pip install -r requirements.txt
    ```
    
    3. Run main.py
    
    ```
    streamlit run main.py
    ```
    """) 

    
def layout():
    selected = option_menu(
            menu_title=None,
            options=["Prediction", 'About'], 
            icons=['📥', '🔄','🔄'], 
            menu_icon="Data Cleaning", 
            default_index=0,
            orientation="horizontal"
            )
    return selected
   
def main():
    # Labels
    label_name = ['Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
              'Cherry healthy', 'Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight',
              'Corn healthy', 'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy', 'Peach Bacterial spot',
              'Peach healthy', 'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight',
              'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight',
              'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites', 'Tomato Target Spot',
              'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']
    
    # Load the model
    try:
        model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')
    except Exception as e:
        st.error("Model Not Found")
    # Add File uploader
    st.header("Upload Image")
    uploaded_file = st.file_uploader("")
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
        normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)
        st.image(image_bytes)
        if st.button("Find Disease"):
            predictions = model.predict(normalized_image)
            # color that resembles a disease
            disease_color = "#FF5733"  
            # Define disease-related icon
            disease_icon = "🤒"  # You can replace it with any disease-related icon

            if predictions[0][np.argmax(predictions)] * 100 >= 80:
                st.markdown(f"<h4 style='text-align: center; color: {disease_color};'>{disease_icon} {label_name[np.argmax(predictions)]} Found {disease_icon}</h4>", unsafe_allow_html=True)
            else:st.write(f"Try Another Image")



if __name__=="__main__":
    option=layout()
    if(option=='Prediction'):
        # User info
        st.info(f"Please input only leaf images of Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, and Tomato. Otherwise, the model may not work as expected.",icon="🚨")
        # main Fun
        main()
    elif(option=='About'):
        st.markdown("<h1 style='text-align: center; color: #38B000;'>💡 About Project 🌿</h1>", unsafe_allow_html=True)
        user_guide()