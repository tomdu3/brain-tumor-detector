import streamlit as st


def page_summary_body():

    st.write('### Quick Project Summary')

    st.info(
        f'**General Information**\n'
        f'* Generally about the brain tumor\n'
        f'* Different causes.\n'
        )

    st.write(
        f'* For additional information, please visit and **read** the '
        f'[Project README file](https://github.com/tomdu3/brain-tumor-detector).')
    

    st.success(
        f'The project has 2 business requirements:\n'
        f'* 1 - The client would like to have a study of the dataset collected at the '
        f'Department of Brain Repair and Rehabilitation\n' 
        f'* 2 - The client would like to have a ML model developed in order to be able to identify the brain tumor  from the MRI scan.'
        )