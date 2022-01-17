import streamlit as st
import os
import glob
import datetime
import tarfile
import urllib.request
from .utils import isEmpty
from .constants import DATASET_PATH

def download_dataset(filename, url, work_dir):
    if not os.path.exists(filename):
        with st.spinner('Downloading flowers17 dataset....'):
            print("[INFO] Downloading flowers17 dataset....")
            filename, _ = urllib.request.urlretrieve(url + filename, filename)
            statinfo = os.stat(filename)
        st.info("[INFO] Succesfully downloaded " + filename + " " + str(statinfo.st_size) + " bytes.")
        print("[INFO] Succesfully downloaded " + filename + " " + str(statinfo.st_size) + " bytes.")
        untar(filename, work_dir)
    else:
        isEmpty(DATASET_PATH)
        st.info('[INFO] Dataset already downloaded')
        
def jpg_files(members):
	for tarinfo in members:
		if os.path.splitext(tarinfo.name)[1] == ".jpg":
			yield tarinfo

def untar(fname, path):
    tar = tarfile.open(fname)
    tar.extractall(path=path, members=jpg_files(tar))
    tar.close()
    st.info("[INFO] Dataset extracted successfully.")
    print("[INFO] Dataset extracted successfully.")



def app():
    st.header('Dataset Preping ')
    flowers17_url  = "http://www.robots.ox.ac.uk/~vgg/data/flowers/17/"
    flowers17_name = "17flowers.tgz"
    train_dir      = "dataset"

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    download_dataset(flowers17_name, flowers17_url, train_dir)
    

    if os.path.exists(train_dir + "\\jpg") and not os.path.exists(train_dir + "\\train"):
        os.rename(train_dir + "\\jpg", train_dir + "\\train")


    # get the class label limit
    class_limit = 17

    # take all the images from the dataset
    image_paths = glob.glob(train_dir + "\\train\\*.jpg")

    # variables to keep track
    label = 0
    i = 0
    j = 80

    # flower17 class names
    class_names = ["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus",
                    "iris", "tigerlily", "tulip", "fritillary", "sunflower", 
                    "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup",
                    "windflower", "pansy"]

    # loop over the class labels
    if not os.path.exists(train_dir + "\\train\\" + class_names[label]):
        with st.spinner('Organizing the dataset'):
            for x in range(1, class_limit+1):
                # create a folder for that class
                
                os.makedirs(train_dir + "\\train\\" + class_names[label])
                    # get the current path
                cur_path = train_dir + "\\train\\" + class_names[label] + "\\"
                
                # loop over the images in the dataset
                for index, image_path in enumerate(image_paths[i:j], start=1):
                    original_path   = image_path
                    image_path      = image_path.split("\\")
                    image_file_name = str(index) + ".jpg"
                    os.rename(original_path, cur_path + image_file_name)
                
                i += 80
                j += 80
                label += 1
        st.info('[INFO] Dataset organized successfully')
    else:
        st.info('[INFO] Dataset already organized')
