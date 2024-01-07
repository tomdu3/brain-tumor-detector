import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_mri_visualizer import page_mri_visualizer_body
from app_pages.page_ml_performance import page_ml_performance_metrics
from app_pages.page_tumor_detector import page_tumor_detector_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body


# Create an instance of the app
app = MultiPage(app_name='Brain Tumor Detector')

# Add your app pages here using .add_page()
app.add_page('Quick Project Summary', page_summary_body)
app.add_page('MRI Visualizer', page_mri_visualizer_body)
app.add_page('Model Performance', page_ml_performance_metrics)
app.add_page('Brain Tumor Detection', page_tumor_detector_body)
app.add_page('Project Hypothesis', page_project_hypothesis_body)
app.run()  # Run the app
