# Landmark_Detection_Asia_Pecific
Landmark Detection Asia Pacific
This repository contains a simple Python script that uses Gradio to build a web app for landmark detection in Asia. The script uses a TensorFlow Hub model to classify images into different landmark categories.

Requirements
Python 3.6 or higher
NumPy
Pandas
Matplotlib
Gradio
PIL.Image
TensorFlow
TensorFlow Hub
Instructions
To run the script, clone the repository and install the required dependencies:

git clone https://github.com/GurucharanSavanth/Landmark_Detection_Asia_Pecific.git
cd Landmark_Detection_Asia_Pecific
pip install -r requirements.txt

Download the training images from https://s3.amazonaws.com/google-landmark/train/images_345.tar and extract them to the images_345 directory.

Then, run the script:

python landmark_detection.py

This will launch a web app in your browser. You can then upload an image and the app will predict the landmark in the image.

Usage
To use the web app, simply upload an image and click the "Classify" button. The app will then predict the landmark in the image and display the result.

Example
Here is an example of how to use the web app:

Upload an image of a landmark.
Click the "Classify" button.
The app will predict the landmark in the image and display the result.
For example, if you upload an image of the Eiffel Tower, the app will predict "Eiffel Tower" as the landmark.

Deployment
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

License
This repository is licensed under the MIT License.
