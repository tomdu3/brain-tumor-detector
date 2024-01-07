import streamlit as st


def page_summary_body():

    st.write('### Quick Project Summary')

    st.info(
        'Brain Tumor Detector is a data science and machine learning project. '
        'The business goal of this project is the differentiation of the '
        'healthy brain and the one with the tumor based on the brain MRI '
        'scan images. The project is realised with the Streamlit Dashboard '
        'and gives to the client a possibility to upload the MRI brain scan '
        'in order to predict the possible tumor diagnosis. The dashboard '
        'offers the results of the data analysis, description and the '
        'analysis of the project\'s hypothesis, and details about the '
        'performance of the machine learning model.'
        )

    st.write(
        '* For additional information, please visit and read the '
        '[Project\'s README file]'
        '(https://github.com/tomdu3/brain-tumor-detector).')

    st.success(
        'The project has 2 business requirements:\n'
        '* 1 - The client would like to have a study of the dataset'
        'collected \n'
        '* 2 - The client would like to have a ML model developed in order '
        'to be able to identify the brain tumor  from the MRI scan.'
        )
