import streamlit as st

def page_project_hypothesis_body():
    st.write('### Project Hypothesis and Validation')

    st.success(
        f'* We believe that the brain tumor on the MRI scan shows consistently '
        f'darker shade of the image and if it\'s in an advanced stage, it show a denser image than the one of a healthy brain. \n\n'
        f'* Average Image, Variability Image and Difference between Averages studies did give some insights, '
        f'since the brain varies according to it\'s size, development and age, it is not possible '
        f'to find clear patterns that could be easily noticeable for the differentiation.'
    )
