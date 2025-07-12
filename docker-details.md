The Dockerfile has been created to ease the local run of the Streamlit app.

To build the Docker image, run the following command in your terminal from the project root directory:
```sh
docker build -t brain-tumor-detector .
 ```


Once the image is built, you can run the Streamlit application using:
```sh
docker run -p 8501:8501 brain-tumor-detector
```

This will map port 8501 from the container to port 8501 on your local machine, allowing you to access the Streamlit app in your browser at `http://localhost:8501`.
