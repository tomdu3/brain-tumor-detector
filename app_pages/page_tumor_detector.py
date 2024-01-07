import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
                                                    load_model_and_predict,
                                                    resize_input_image,
                                                    plot_predictions_probabilities  # noqa
                                                    )


def page_tumor_detector_body():
    st.write('### Brain Tumor Detection')

    st.info(
        '* The client would like to be able to predict the presence of tumors '
        'in a given brain MRI scan.'
        )

    st.write(
        '* The training was made on the dataset from Kaggle. So, the test '
        'image could be taken from the this link: '
        '[Kaggle Dataset]'
        '(https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor/data).'
        )

    st.write('---')

    images_buffer = st.file_uploader(
        'Upload brain MRI scan. You may select more than one.',
        type=['png', 'jpg'], accept_multiple_files=True)

    if images_buffer is not None:
        df_report = pd.DataFrame([])
        for image in images_buffer:

            img_pil = (Image.open(image))
            st.info(f'Brain MRI Scan Sample: **{image.name}**')
            img_array = np.array(img_pil)
            st.image(img_pil,
                     caption=f'Image Size: {img_array.shape[1]}px width x '
                             f'{img_array.shape[0]}px height')

            version = 'v4'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img,
                                                            version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            df_report = df_report.append(
                {'Name': image.name, 'Result': pred_class}, ignore_index=True)

        if not df_report.empty:
            st.success('Analysis Report')
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report),
                        unsafe_allow_html=True)
