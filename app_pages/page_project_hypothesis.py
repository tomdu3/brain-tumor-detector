import streamlit as st


def page_project_hypothesis_body():
    st.write('### Project Hypothesis and Validation')

    st.success(
        "1. There's a strong conviction that there could be a way to visually "
        "observe and notice the difference in brain MRI scans between a "
        "healthy brain and the one with the tumor. Being in low resolution, "
        "the filtering of the MRI scans and the comparison between the "
        "average scan of the tumor and the healthy brain scan should show "
        "the visible shade difference.\n"
        "2. The deep learning model with convolutional neural network (CNN) "
        "should be able to accurately architecture should be able to "
        "accurately classify the unseen data of the brain MRI images as "
        "tumor or non-tumor. Data augmentation techniques will help improve "
        "model generalization."
    )
    st.write('---')
    st.warning(
        'However, upon validation, the ML model faced challenges:\n\n'
        '1. **Ambiguity in Visual Differentiation:** In instances where the '
        'difference between healthy and tumor-affected scans wasn\'t '
        'pronounced, the model struggled to make accurate distinctions.\n\n'
        '2. **Insufficient Performance Metrics:** The accuracy and F1 scores '
        'of the model did not meet the predefined thresholds, indicating that '
        'the model was notthe model was not yet ready for approval in its '
        'current state.\n\n'
        'As a result, while the project showed promise, further refinement '
        'and testing of the ML model are necessary to achieve the desired '
        'level of accuracy in differentiating between healthy and '
        'tumor-affected brain MRI scans.'
    )
