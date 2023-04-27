### Loading Packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import cv2 

### Loading packages for app development
import streamlit as st
from PIL import Image, ImageOps

### Parameters
size_ = (100, 100)
imageUploaded = False


#*********************************
#**** App Config
#*********************************
st.set_page_config(
    page_title="WoodAnomaly",
    layout="wide",
    initial_sidebar_state="expanded"
)

#*********************************
#**** Importing custom CSS
#*********************************
def customCss(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

customCss("custom.css")

#*********************************
#**** View Sidebar
#*********************************
uploadedImage = st.sidebar.file_uploader("Upload Image", accept_multiple_files=False)
st.sidebar.markdown('<br>', unsafe_allow_html=True)
submit = st.sidebar.button('Detect Anomaly', key=None, help=None, on_click=None)


#********************************
#**** Main body
#********************************
bodyCol1, bodyCol2 = st.columns(2)
## Displaying actual image
if submit:
    #imageData = Image.open(uploadedImage)

    img = np.array(Image.open(uploadedImage))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imageEdge = cv2.Canny(img, 30, 50)
    st.text(imageEdge.shape)
    #cv2.imwrite('img.png', imageEdge)

    anoImageList = []
    #st.text(np.array(Image.open('img.png')).shape)
    #img = np.array(Image.open('img.png').resize((128, 128)))
    img = cv2.resize(imageEdge, (128, 128), interpolation= cv2.INTER_LINEAR)
    imgArray = img.reshape((1, 128*128))
    imgArray = imgArray/255
    anoImageList.append(imgArray.tolist()[0])

    ##### Calling mmodel for generating predictions
    ### Loading saved model
    jFile = open('cnn_autoencoder_24032023'+'.json', 'r')
    loaded_model_json = jFile.read()
    jFile.close()
    autoencoder = model_from_json(loaded_model_json)
    autoencoder.load_weights('cnn_autoencoder_24032023'+'.h5')
    ### Data preparation for modeling
    anoImageDf = pd.DataFrame(anoImageList)
    xTestCnnCat = np.array(anoImageDf)
    print(len(xTestCnnCat))
    xTestCnnCat= np.reshape(xTestCnnCat, (len(xTestCnnCat), 128, 128, 1))


    ## Making predictions
    reconstructions = autoencoder.predict(xTestCnnCat)
    test_loss = tf.keras.losses.mae(reconstructions.reshape((1, 128*128)), xTestCnnCat.reshape((1, 128*128)))
    testPred = [1 if i>0.19452723240978956/3.35 else 0 for i in test_loss]

    ### Displaying content
    bodyCol1.markdown('''<div style='background-color: rgb(242, 25, 74); padding: 5px; border: 2px solid rgb(242, 25, 74); color: white;'> Acutal Image (Color) </div>''', unsafe_allow_html=True)
    bodyCol1.image(Image.open(uploadedImage).resize((200, 200)), use_column_width='always')
    bodyCol2.markdown('''<div style='background-color: rgb(242, 25, 74); padding: 5px; border: 2px solid rgb(242, 25, 74); color: white;'> Acutal Image (GreyScaled) </div>''', unsafe_allow_html=True)
    bodyCol2.image(ImageOps.grayscale(Image.open('img.png').resize((200, 200))), use_column_width='always')

    st.markdown('''<h4> Prediction... </h4>''', unsafe_allow_html=True )
    
    if testPred[0] == 0:
        st.balloons()
        st.success('Uploaded image is Non-Anamolous', icon="✅")
    else:
        st.error('Uploaded image is Anamolous', icon="🚨")

else:
    st.warning('Please upload image to get recommendations', icon="⚠️")










