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
- The client is interested in having an analysis of the visual difference between the MRI brain scan of healthy and brain with tumor. The analysis should provide: the average image and variablity per label in the data set.
- The client is interested in having a functional and reliable ML model that could predict the presence or absence of the tumor from the image of the MRI brain scan. For the realisation of this business objective, a deep learning pipeline should be developed with the binary classification of the MRI images. The said pipeline should be also deployed.
- The Streamlit Dashboard will be developed that will finally serve as a platform for the presentation of the results of first two business objectives, together with the interactive implementation of the prediction of the unseen MRI image.


## Hypothesis and how to validate?
- Hypothesis 1: The deep learning model with convolutional neural network (CNN) architecture will be able to accurately classify brain images as tumor or non-tumor.

- Hypothesis 2: Data augmentation techniques will help improve model generalization.

## The rationale to map the business requirements to the Data Visualizations and ML tasks
* List your business requirements and a rationale to map them to the Data Visualizations and ML tasks
**Business Requirement 1**: Data Visualization
**Business Requirement 2**: Classification

- Accuracy: Visualisations should make the model's performance metrics comprehensible. We will plot learning curves to monitor training progress and use confusion matrices for a detailed breakdown of classification results.

Interpretability: Visualizations, especially heatmaps, will provide insight into how the model is making predictions. It is aligned with the interpretability requirement.

Scalability: We will analyze the model's performance on varying sizes of datasets using visualizations to ensure it scales efficiently.

Speed: Monitoring the model's inference time will ensure it meets the speed requirement.

Privacy: This will be ensured through data anonymization, which will be part of the data handling and model deployment processes.

## ML Business Case
* In the previous bullet, you potentially visualized an ML task to answer a business requirement. You should frame the business case using the method we covered in the course 


## Dashboard Design
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other item that your dashboard library supports.
* Later, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but subsequently you used another plot type).



## Unfixed Bugs
* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
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

