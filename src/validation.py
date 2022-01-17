from src.trainning.train_pipeline import TrainingPipeline
import streamlit as st
from PIL import Image



def app():
    tp=TrainingPipeline()
    with st.spinner('Models comparison'):
        tp.cross_validation(plot_name='new')
        st.image(Image.open("output\\plot\\new.png"))