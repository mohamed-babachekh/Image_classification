from numpy.core.fromnumeric import sort
import streamlit as st
from src.constants import FIXED_SIZE, MODELS,TEST_PATH,TRAIN_PATH,PREDICTION_PATH
from src.features import extract_features, scale_features
from src.utils import change_output_format, infere
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt

train_labels = os.listdir(TRAIN_PATH)
# sort the training labels
train_labels.sort()

def app():
    st.subheader('Predicting flower class')
    with st.form('inference'):
        uploaded_image=st.file_uploader('import your image', 
                         type=['png', 'jpg'], 
                         accept_multiple_files=False
                        )
        selected_model=st.selectbox('predict using ',
                                    options= MODELS ,
                                    format_func= change_output_format,
                                    index=0)
        
        submitted = st.form_submit_button("Predict")
    
    if submitted:
    
        with Image.open(uploaded_image) as im:  
            im.save(TEST_PATH+"\\"+uploaded_image.name,"png",bitmap_format='png')
        st.write(uploaded_image)
 
        image = cv2.imread(TEST_PATH+"\\"+uploaded_image.name)
        image=cv2.resize(image,FIXED_SIZE)
        global_feature=extract_features(image)
       
        prediction = infere(global_feature.reshape(1,-1),selected_model[0])
        cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 3)
        # display the output image          
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plot_path = str(PREDICTION_PATH)
        plt.savefig(plot_path,bbox_inches='tight',pad_inches=0.33)
        
        st.image(Image.open("output\\plot\\infere.png"))
        st.write(train_labels[prediction])
 

