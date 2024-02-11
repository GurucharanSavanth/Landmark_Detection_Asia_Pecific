import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import gradio as gr
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
from gradio import components
import streamlit as st

## Downlode https://s3.amazonaws.com/google-landmark/train/images_345.tar
## You can use different Sources Eg. Local Path or online dataset model and csv model as well  
# Load model and label map
classifier = tf.keras.Sequential([hub.KerasLayer(TF_MODEL_URL, input_shape=IMAGE_SHAPE + (3,), output_key="predictions:logits")])
df = pd.read_csv(LABEL_MAP_URL)
label_map = dict(zip(df.id, df.name))

# Function to process and predict image
def process_and_predict_image(img_path):
    img = Image.open(img_path).resize(IMAGE_SHAPE)
    img_array = np.array(img) / 255.0
    img_array = img_array[np.newaxis, ...]
    result = classifier.predict(img_array)
    predicted_label = label_map[np.argmax(result)]
    return img_array, predicted_label

# Streamlit App
st.title('Landmark Prediction App')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize(IMAGE_SHAPE)
    st.image(image, caption='Input Image', use_column_width=True)
    processed_image, prediction = process_and_predict_image(uploaded_file)
    st.write(f"Predicted Landmark: {prediction}")

    # Plot the image with prediction
    fig, ax = plt.subplots()
    ax.imshow(np.squeeze(processed_image))
    ax.set_title(f'Predicted landmark: {prediction}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)

# Gradio Interface
def classify_image(image):
    img = Image.fromarray(image.astype('uint8'), 'RGB').resize(IMAGE_SHAPE)
    img_array = np.array(img) / 255.0
    img_array = img_array[np.newaxis, ...]
    prediction = classifier.predict(img_array)
    return label_map[np.argmax(prediction)]

image_input = gr.components.Image()
label_output = gr.components.Label(num_top_classes=1)

gr.Interface(
    classify_image,
    inputs=image_input,
    outputs=label_output
).launch(share=True)
# to run the program using streamlit
# streamlit run colab_kernel_launcher.py
