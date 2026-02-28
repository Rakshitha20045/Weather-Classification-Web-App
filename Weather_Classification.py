from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras import layers
import numpy as np
from PIL import Image, ImageOps

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Custom object for loading model without 'groups' argument in DepthwiseConv2D
def custom_depthwise_conv2d(*args, **kwargs):
    if 'groups' in kwargs:
        del kwargs['groups']  # Remove the groups argument
    return DepthwiseConv2D(*args, **kwargs)

# Load the model with the custom object
model = load_model("D:\svasti vector solution\Weather Classification_image_model\keras_model.h5", compile=False, custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d})

# Load the labels
class_names = open("D:\svasti vector solution\Weather Classification_image_model\labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("D:\svasti vector solution\Weather Classification_image_model\dataset\sunrise\sunrise105.jpg").convert("RGB")

# Resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predict using the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")  # Skipping the newline character at the beginning of class_name
print("Confidence Score:", confidence_score)
