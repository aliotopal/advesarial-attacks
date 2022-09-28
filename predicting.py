
import numpy as np
from keras.applications.vgg16 import preprocess_input, VGG16, decode_predictions
from keras_preprocessing.image import img_to_array, load_img



model = VGG16(weights='imagenet')

img = np.load("adversPGD.npy")
# img = load_img("adversPGD.png", target_size=(224, 224))
# img = img_to_array(img)


image = img.reshape((1, 224, 224, 3))
yhat = model.predict(preprocess_input(image))
label0 = decode_predictions(yhat,1000)
label1 = label0[0][0]
print("The image isssss: %s  %.4f%%" % (label1[1], label1[2]))
print(label0)
print(label1[1], label1[2])