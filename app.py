import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts


app = MultiPage(app_name="Brain Tumor Detector")  # Create an instance of the app

# Add your app pages here using .add_page()


app.run()  # Run the app