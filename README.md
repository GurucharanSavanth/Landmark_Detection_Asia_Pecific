## Landmark Detection Asia Pacific

This repository contains a simple Python script that uses Gradio to build a web app for landmark detection in Asia. The script uses a TensorFlow Hub model to classify images into different landmark categories.

**Features:**

* Accurately detects over 1,000 landmarks in Asia
* Easy to use web app interface
* Can be deployed to cloud services like Gradio Hub or Heroku

**Requirements:**

* Python 3.6 or higher
* NumPy
* Pandas
* Matplotlib
* Gradio
* PIL.Image
* TensorFlow
* TensorFlow Hub
* streamlit

**Instructions:**

1. Clone the repository:
``git clone https://github.com/GurucharanSavanth/Landmark_Detection_Asia_Pecific.git``

2. Install the required dependencies:
 ``pip install -r requirements.txt``
3. Download the training images from https://s3.amazonaws.com/google-landmark/train/images_345.tar and extract them to the images_345 directory.

4. Run the web app:
```python python main.py```

This will launch a web app in your browser. You can then upload an image and the app will predict the landmark in the image.

**Usage:**

To use the web app, simply upload an image and click the "Classify" button. The app will then predict the landmark in the image and display the result.

**Example:**

Here is an example of how to use the web app:

Upload an image of a landmark.
Click the "Classify" button.
The app will predict the landmark in the image and display the result.
For example, if you upload an image of the Eiffel Tower, the app will predict "Eiffel Tower" as the landmark.

**NOTE : This Program has Limited Dataset sourced from the kaggle so the predection is only 85% in Worst case and 87% in Best case  Predection**

**Deployment:**

To deploy the web app, you can use a service like Gradio Hub or Heroku.

To deploy the web app to Gradio Hub, follow these steps:

Sign up for a Gradio Hub account.
Create a new project.
Upload the landmark_detection.py script to your project.
Click the "Deploy" button.
To deploy the web app to Heroku, follow these steps:

Create a Heroku account.
Create a new Heroku app.
Deploy the landmark_detection.py script to your Heroku app.
Once you have deployed the web app, you can share the link with others so that they can use it.
