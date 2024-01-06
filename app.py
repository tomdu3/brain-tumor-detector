import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_mri_visualizer import page_mri_visualizer_body
from app_pages.page_ml_performance import page_ml_performance_metrics


app = MultiPage(app_name='Brain Tumor Detector')  # Create an instance of the app

# Add your app pages here using .add_page()
app.add_page('Quick Project Summary', page_summary_body)
app.add_page('MRI Visualizer', page_mri_visualizer_body)
app.add_page('Model Performance', page_ml_performance_metrics)
app.run()  # Run the app