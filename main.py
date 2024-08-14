import streamlit as st
import tensorflow as tf
import numpy as np

MODEL_PATH = r"Model\training_model.keras"
HOME_IMAGE = r"Images\Banner.jpg"

st.set_page_config(
    page_title="Fruits & Veg Prediction",
    page_icon="🤖",
    layout='wide',
    menu_items={
        'Get Help': 'https://github.com/Nasif-Azam',
        'Report a bug': 'mailto:nasifazam07@gmail.com',
        'About': "### Fruits & Veg Prediction\nThis app predicts fruits and vegetables using AI."
    }
)
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Page:", ["Home", "About Project", "Prediction"])


def HomePage():
    st.header("Fruits & Vegetables Recognition System")
    st.image(HOME_IMAGE)
    st.markdown(
        """
        **Using the CNN to build the recognition system**        
        ### About Me
        **<i class="fa fa-user"></i> Name:** Nasif Azam\n
        **<i class="fa fa-envelope"></i> Email:** nasifazam07@gmail.com\n
        **<i class="fa fa-mobile-alt"></i> Phone:** +880-1533903305\n
        **<i class="fa fa-map-marker-alt"></i> Address:** Mirpur, Dhaka, Bangladesh
        
        <a href="https://www.facebook.com/md.nasif850" target="_blank" style="margin-right: 10px;"><i class="fab fa-facebook fa-2x"></i></a>
        <a href="https://github.com/Nasif-Azam" target="_blank" style="margin-right: 10px; color:black;"><i class="fab fa-github fa-2x"></i></a> 
        <a href="https://www.linkedin.com/in/nasif-azam-9aa2331a0/" target="_blank" style="margin-right: 10px; color:sky;"><i class="fab fa-linkedin fa-2x"></i></a> 
        <a href="https://www.hackerrank.com/profile/Nasif_Azam" target="_blank" style="margin-right: 10px; color:green;"><i class="fab fa-hackerrank fa-2x"></i></a> 
        <a href="https://www.kaggle.com/nasifazam" target="_blank" style="margin-right: 10px; color:blue;"><i class="fab fa-kaggle fa-2x"></i></a>  
        
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """, unsafe_allow_html=True
    )


def AboutPage():
    st.header("About Project")
    st.markdown("""
     ### Dataset
     This dataset encompasses images of various fruits and vegetables, providing a diverse collection for image recognition tasks. The included food items are:\n
     - **10 Fruits:** Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango.\n
     - **26 Vegetables:** Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant.
    ### Content
     The dataset is organized into three main folders:
    - **Train:** Contains 100 images per category.
    - **Test:** Contains 10 images per category.
    - **Validation:** Contains 10 images per category.
    Each of these folders is subdivided into specific folders for each type of fruit and vegetable, containing respective images.
    ### Data Collection
    The images in this dataset were sourced using Bing Image Search for a personal project focused on image recognition of food items. The creator does not hold the rights to any of the images included in this dataset. If you are the owner of any image and have concerns regarding its use, please contact the creator to request its removal. The creator will promptly comply with any such requests to ensure all legal obligations are met.
    ### Disclaimer
    Users of this dataset are responsible for ensuring that their use of the images complies with applicable copyright laws and regulations. The creator assumes no responsibility for any legal issues that may arise from the use of this dataset. It is recommended to use the dataset for educational and non-commercial purposes only and to seek legal counsel if you have specific concerns about copyright compliance.
     """, unsafe_allow_html=True)


def PredictionPage():
    st.header("Prediction")
    test_image = st.file_uploader("Choose an Image: ")
    if test_image:
        st.image(test_image, width=280, use_column_width=False)
        if (st.button("Predict")):
            st.balloons()
            result_index = Model_Prediction(test_image)
            with open("Labels.txt") as f:
                content = f.readlines()
            # st.write(content)
            label = []
            for i in content:
                label.append(i[:-1])
            # st.write(label)
            st.write("Model Predicting... ")
            st.success(label[result_index])
    else:
        st.markdown('<p style="color:red;">Upload A Image First!!</p>', unsafe_allow_html=True)


def Model_Prediction(test_image):
    model = tf.keras.models.load_model(MODEL_PATH)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single img to batch
    prediction = model.predict(input_arr)
    return np.argmax(prediction)


if app_mode == "Home":
    HomePage()
elif app_mode == "About Project":
    AboutPage()
else:
    PredictionPage()
