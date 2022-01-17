from src.trainning.train_pipeline import TrainingPipeline
import streamlit as st
import time
from src.utils import change_output_format
from src.constants import MODELS




def app():
    st.subheader('Training models')


    with st.form("my_form"): 
        model_selected= st.selectbox('model to be trained',
                                    options= MODELS ,
                                    format_func= change_output_format,
                                    index=0)
                    
        train = st.form_submit_button("Train Model")
    if train:
        with st.spinner('Training model, please wait...'):
            time.sleep(1)
            try:
                tp=TrainingPipeline()
                tp.train(model_name=model_selected[0])

                accuracy, f1 = tp.get_model_perfomance()
                
                col1, col2 = st.columns(2)

                col1.metric(label="Accuracy score", value=str(round(accuracy,5)))
                col2.metric(label="F1 score", value=str(round(f1,5)))
            except Exception as e:
                st.error('Failed to train model!')
                st.exception(e)
                