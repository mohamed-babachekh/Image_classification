from multiOptions import Multioption
import streamlit as st
from src import flower_dataset,feature_extraction, inference,train, validation



st.title("Image Classification")
st.sidebar.title("Project Steps")

app = Multioption()

app.add_option('Dataset download',flower_dataset.app)
app.add_option('Feature extraction',feature_extraction.app)
app.add_option('Training',train.app)
app.add_option('Cross Validation',validation.app)
app.add_option('Prediction',inference.app)


app.run()
