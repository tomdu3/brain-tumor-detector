![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

# Brain Tumor Detector

Brain Tumor Detector is a data science and machine learning project that is a 5th and final Project of the Code Institute's Bootcamp in Full Stack Software Development with specialization in Predictive Analytics.
The business goal of this project is the differentiation of the healthy brain and the one with the tumor based on the brain MRI scan images. The project is realised with the [Streamlit Dashboard](## TODO - heroku link) and gives to the client a possibility to upload the MRI brain scan in order to predict the possible tumor diagnosis. THe dashboard offers the results of the data analysis, description and the analysis of the project's hypothesis, and details about the performance of the machine learning model.
The project includes a series of Jupyter Notebooks that represent a pipleine = importation and cleaning  of the data, data visualization, development and evaluation of the deep learning model.

## Dataset Content
The dataset is **Brain Tumor** dataset from [Kaggle](https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor/data)

This is a brain tumor feature dataset including five first-order features and eight texture features with the target level (in the column Class).

- First Order Features
    - Mean
    - Variance
    - Standard Deviation
    - Skewness
    - Kurtosis

- Second Order Features
    - Contrast
    - Energy
    - ASM (Angular second moment)
    - Entropy
    - Homogeneity
    - Dissimilarity
    - Correlation
    - Coarseness 

Image column defines image name and Class column defines either the image has tumor or not (1 = Tumor, 0 = Non-Tumor)

## Business Requirements
The (fictitious) Department of Brain Repair and Rehabilitation of the Health Institute in London, UK
The primary objective of this project is to develop a machine learning model for the early detection of brain tumors from medical images. The model should assist medical professionals in making quicker and more accurate diagnoses, and the patients should benefit from the earlier detection and the tempestive and appropriate treatment planning.

Key Stakeholders, therefore should be:
    - Medical professionals
    - Patients
    - Hospitals and healthcare facilities

Requirements:

- Accuracy: The model should have a high accuracy rate in classifying brain images as either tumor (1) or non-tumor (0).
- Interpretability: The model should provide some insight into the prediction process and the relevant feature importance in that process, so the medical professionals could understand the relevant discoveries.
- Scalability: The solution should be scalable to handle a large volume of brain images from various sources.
- Speed: The model should be able to make predictions in real-time so that the reliable quick diagnosis could be make.
- Privacy: The meticulous attention should be given in the data collection in order to guarantee the patient's anonymity and consent for the data usage.

In short, the project businsess objectives are as follows:
- (1) The client is interested in having an analysis of the visual difference between the MRI brain scan of healthy and brain with tumor. The analysis should provide: the average image and variability per label in the data set.
- (2) The client is interested in having a functional and reliable ML model that could predict the presence or absence of the tumor from the image of the MRI brain scan. For the realisation of this business objective, a deep learning pipeline should be developed with the binary classification of the MRI images. The said pipeline should be also deployed.
- The Streamlit Dashboard will be developed that will finally serve as a platform for the presentation of the results of first two business objectives, together with the interactive implementation of the prediction of the unseen MRI image.


## Hypothesis and how to validate?

The project's initial hypothesis was for each business objective as follows:
- There's a strong conviction that there could be a way to visually observe and notice the difference in brain MRI scans between a healthy brain and the one with the tumor. Being in low resolution, the filtering of the MRI scans and the comparison between the average scan of the tumor and the healthy brain scan should show the visible shade difference.
-  The deep learning model with convolutional neural network (CNN) architecture should be able to accurately classify the unseend data of the brain MRI images as tumor or non-tumor. Data augmentation techniques will help improve model generalization.

- The validation od these hypotheses should be made through the graphical evaluation of the generated model, abd throug hthe testing. The model should include the validation of its accuracy and loss between epochs, and finally through a confusion matrix.
- Upon the validation of these two hypotheses, the client should be able to use the conventional image data analysis and the ML model of this project in order to differentiate with high accuracy the presence or not of the tumor by the means of the brain MRI scan.



## The rationale to map the business requirements to the Data Visualizations and ML tasks

- Accuracy: Visualisations should make the model's performance metrics comprehensible. We will plot learning curves to monitor training progress and use confusion matrices for a detailed breakdown of classification results.
Interpretability: Visualizations will provide insight into how the model is making predictions. It is aligned with the interpretability requirement.
Scalability: We will analyze the model's performance on varying sizes of datasets using visualizations to ensure it scales efficiently.
Speed: Monitoring the model's inference time will ensure it meets the speed requirement.
Privacy: This will be ensured through data anonymization, which will be part of the data handling and model deployment processes.

Business Requirement 1: Data Visualization
- As a client, I can navigate easily through an interactive dashboard so that I can view and understand the data.
- As a client, I can view visual graphs of average images,image differences and variabilities between MRI of a healthy brain and the one of the tumor, so that I can identify which is which more easily.
- As a client, I can view an image montage of the MRI's of the healthy brain and the one with tumor, so I can make the visual differentiation.

Business Requirement 2: Classification
- As a client, I can use a machine learning model so that I can obtain a class prediction on a new unseen brain MRI image.
- As a client, I can upload image(s) of the brain MRI scans to the dashboard so that I can run the ML model and an immediate accurate prediction of the posible brain tumor.
- As a client, I can save model predictions in a timestamped CSV file so that I can have a documented history of the made predictions.

## ML Business Case

- The client, the Department of Brain Repair and Rehabilitation of the Health Institute in London, UK, is focused on accurately predicting from a given medical image whether a brain tumor is present. This business objective will be achieved through the development and deployment of a TensorFlow deep learning pipeline, trained on a dataset of brain images classified as either having a tumor or not.
- This TensorFlow pipeline will employ a convolutional neural network (CNN), a type of neural network particularly effective at identifying patterns and key features in image data, utilizing convolution and pooling layer pairs.
- The ultimate goal of this machine learning pipeline is a binary classification model. The desired outcome is a model capable of successfully distinguishing brain images as either having a tumor or being tumor-free.
- The model's output will be a classification label indicating the presence or absence of a brain tumor, based on the probability calculated by the model.
- Upon generating the outcome, the following heuristic will be applied: Brain images identified as having a tumor will be flagged for further medical review and potential treatment planning, while those classified as tumor-free may undergo additional checks as per medical protocols.
- The primary metrics for evaluating the success of this machine learning model will be overall model accuracy (measured by the F1 score) and recall for correctly identifying brain images with tumors.
- The reasonable accuracy threshold shoud be set very high by the stakeholder, but the dataset provided could be quite limiting. A model with high accuracy will be crucial in ensuring reliable diagnoses, thereby improving patient outcomes and optimizing the use of medical resources.
- High recall in detecting brain tumors is critical, as the cost of not identifying a present tumor (false negatives) is significantly higher than incorrectly identifying a tumor in a healthy brain (false positives). The preliminary threshold for recall should be also reasonabely high, but the dataset could be a limiting factor.
- Therefore, a successful model for this project is one that achieves an F1 score of 0.95 or higher and a recall rate for detecting brain tumors of 0.98 or higher, aligning with the critical nature of accurate and reliable medical diagnoses.


## Dashboard Design
- This project is presented through a Streamlit dashboard web application that consists of five app pages. The client can effortlessly navigate through these pages using the interactive menu located on the left side of the page, as depicted below.
- **Quick Project Summary** - The homepage of the project provides a fundamental overview of the business process that motivates the creation of this project, in addition to providing links to additional documentation.
    - TODO - screenshot Dashboard and menu

- **MRI Visualizer** - The first business objective of the project is addressed by the MRI Visualizer page, which focuses on Data Analysis. This page includes plots that can be toggled on and off using the built-in toolbar. Examples of these plots are provided below.

        - TODO - screenshot plotse
Additionally, this app page offers a tool for creating image montages. Users can select a label class (tumor or non-tumor) and view a montage generated through graphical presentation of random validation set images.

        - TODO - screenshot Montage
- **Model Performance** - The dataset size and label frequencies, which indicate the initial imbalance of the target, are documented on this page. Additionally, the history and evaluation of the project's machine learning model are provided. The paired graphs display the validation loss and accuracy per epoch, showcasing the model's progress over time. Furthermore, a confusion matrix illustrating the predicted and actual outcomes for the test set is presented.

        - TODO - screenshot Montage

- **Brain Tumor Detector** - tool fulfills the second ML business objective of the project. It provides access to the original raw dataset, allowing users to download MRI brain scans. These images can then be uploaded to receive a class prediction output generated by the model.

        - TODO - screenshot of the app
- Here are some examples of the outputs, namely, a binary class prediction, a graphical representation showing percentages, and the option to download the output DataFrame as a CSV file.
        
        - TODO - screenshots of the outputs
- **Project Hypothesis** -
This application page showcases written documentation of the project's hypotheses and analysis of the findings, demonstrating their alignment with the aforementioned hypotheses. The contents is similar to the one in this documentation.

        - TODO screenshots of the app


## Unfixed Bugs
* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment
### Heroku

* The App live link is: [https://brain-tumor-detector-e5d30222dbc4.herokuapp.com/](https://brain-tumor-detector-e5d30222dbc4.herokuapp.com/)
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.


## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries you used in the project and provide an example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open-Source site
- The images used for the gallery page were taken from this other open-source site



## Acknowledgements (optional)
* Thank the people that provided support through this project.

