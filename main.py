import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import gradio as gr
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
from gradio import components
## Downlode https://s3.amazonaws.com/google-landmark/train/images_345.tar
## You can use different Sources Eg. Local Path or online dataset model and csv model as well  
TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'
IMAGE_SHAPE = (321, 321)

classifier = tf.keras.Sequential([hub.KerasLayer(TF_MODEL_URL,
                                                 input_shape=IMAGE_SHAPE + (3,),
                                                 output_key="predictions:logits")])

df = pd.read_csv(LABEL_MAP_URL)
label_map = dict(zip(df.id, df.name))

img_loc = [ "C:/Users/savan/pythonProject/pythonProject/Computer_Vision_V2/images_345/b/0/b/b0b202c05c722f59.jpg","C:/Users/savan/pythonProject/pythonProject/Computer_Vision_V2/images_345/b/0/b/b0b2045cb474a478.jpg","C:/Users/savan/pythonProject/pythonProject/Computer_Vision_V2/images_345/b/0/b/b0b207bb01b5e129.jpg","C:/Users/savan/pythonProject/pythonProject/Computer_Vision_V2/images_345/b/0/b/b0b20eaedbf58be5.jpg","C:/Users/savan/pythonProject/pythonProject/Computer_Vision_V2/images_345/b/0/b/b0b215c27841e2fa.jpg","C:/Users/savan/pythonProject/pythonProject/Computer_Vision_V2/images_345/b/0/b/b0b218017d91e3aa.jpg","C:/Users/savan/pythonProject/pythonProject/Computer_Vision_V2/images_345/b/0/b/b0b21fec1b0afc34.jpg","C:/Users/savan/pythonProject/pythonProject/Computer_Vision_V2/images_345/b/0/b/b0b222ba2cdde805.jpg","C:/Users/savan/pythonProject/pythonProject/Computer_Vision_V2/images_345/b/0/b/b0b22b81544ae2fe.jpg"]
for img_loc in img_loc:
    img = Image.open(img_loc).resize(IMAGE_SHAPE)

    print(img)


img = np.array(img) / 255.0
print(img.shape)

img = img[np.newaxis, ...]
print(img.shape)

result = classifier.predict(img)
print(result)

var = label_map[np.argmax(result)]
class_names = list(label_map.values())
print(var)
print(class_names)

## Squeeze the image array to remove the batch dimension
img = np.squeeze(img)

# Plot the image
fig, ax = plt.subplots()
ax.imshow(img)

# Plot the predicted landmark
landmark = label_map[np.argmax(result)]
ax.scatter(landmark[0], landmark[1], color='red', marker='o')

# Set the title and labels
ax.set_title('Predicted landmark: {}'.format(landmark))
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Show the plot
plt.show()

def classify_image(image):
    img = np.array(image) / 255.0
    img = img[np.newaxis, ...]
    prediction = classifier.predict(img)
    return label_map[np.argmax(prediction)]

image = components.Image(shape=(321, 321))
label = components.Label(num_top_classes=1)

gr.Interface(
    classify_image,
    inputs=image,
    outputs=label,
).launch(Debug=True)


'''You can use the 
gr.Interface(
    classify_image,
    inputs=image,
    outputs=label,
).launch(share=True) for Generally sharing the link which can be used as well 
and 

(Debug=True) for debugging purpous 


'''
