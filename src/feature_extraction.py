import streamlit as st
import os 
import numpy as np
import cv2 
import h5py
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from src.constants import FIXED_SIZE, H5_DATA,H5_DATA,H5_LABELS, H5_LABELS,IMAGE_PER_CLASS,TRAIN_PATH
from src.features import fd_haralick,fd_hu_moments,fd_histogram
from src.features import extract_features,scale_features
def app():

    st.header('Feature Extraction')
    if not os.path.exists(H5_DATA):
        # get the training labels
        train_labels = os.listdir(TRAIN_PATH)

        # sort the training labels
        train_labels.sort()
        print(train_labels)

        # empty lists to hold feature vectors and labels
        global_features = []
        labels          = []

        #progress bar 
        
        col1, col2 = st.columns([2,1])
        with col1:
            st.header("Processed Folders")
        with col2:
            st.header("Progress")
        # loop over the training data sub-folders
        for training_name in train_labels:
            # join the training data path and each species training folder
            dir = os.path.join(TRAIN_PATH, training_name)

            # get the current training label
            current_label = training_name
            # loop over the images in each sub-folder
            with col1:
                st.caption(current_label)   
            with col2:
                pr_bar= st.progress(0)
                for x,percentage_complete in zip(range(1,IMAGE_PER_CLASS+1),range(80)):
                    pr_bar.progress(percentage_complete+21)
                    
                    # get the image file name
                    file = dir + "/" + str(x) + ".jpg"

                    # read the image and resize it to a fIXEDd-size
                    image = cv2.imread(file)
                    image = cv2.resize(image, FIXED_SIZE)

                
                    global_feature = extract_features(image)

                    # update the list of labels and feature vectors
                    labels.append(current_label)
                    global_features.append(global_feature)
                    
                    
            


            print("[STATUS] processed folder: {}".format(current_label))

        st.info("[STATUS] completed Global Feature Extraction...")
        print("[STATUS] completed Global Feature Extraction...")

        # get the overall feature vector size
        print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

        # get the overall training label size
        print("[STATUS] training Labels {}".format(np.array(labels).shape))

        # encode the target labels
        targetNames = np.unique(labels)
        print(targetNames)
        le          = LabelEncoder()

        #PAY AATENTION TO LABELS
        target      = le.fit_transform(labels)
        print("[STATUS] training labels encoded...")

        # scale features in the range (0-1)
        rescaled_features = scale_features(global_features)
        print("[STATUS] feature vector normalized...")

        print("[STATUS] target labels: {}".format(target))
        print("[STATUS] target labels shape: {}".format(target.shape))

        # save the feature vector using HDF5
        h5f_data = h5py.File(H5_DATA, 'w')
        h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

        h5f_label = h5py.File(H5_LABELS, 'w')
        h5f_label.create_dataset('dataset_1', data=np.array(target))

        h5f_data.close()
        h5f_label.close()

        print("[STATUS] end of training..")   
        
    else:
        st.info('Features already extracted and saved')