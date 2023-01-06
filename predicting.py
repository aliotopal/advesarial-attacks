
import numpy as np
import tensorflow as tf
from art.estimators.classification import TensorFlowV2Classifier
from keras.applications.vgg16 import preprocess_input, VGG16, decode_predictions
from keras_preprocessing.image import img_to_array, load_img

mean = [0.485*255, 0.456*255, 0.406*255]
std = [0.229*255, 0.224*255, 0.225*255]

model = VGG16(include_top=True, weights='imagenet', classifier_activation='softmax')

loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
kclassifier = TensorFlowV2Classifier(model = model,
                                     nb_classes=1000,
                                     loss_object=loss,
                                     input_shape=(224,224,3),
                                     clip_values=(0.0,1.0),
                                     # preprocessing=(0.5,0.5)
                                     )
img = np.load("adversPGD.npy")
# img = load_img("adversPGD.png", target_size=(224, 224))
# img = img_to_array(img)
#
#
# image = img.reshape((1, 224, 224, 3))


yhat = model.predict(img)
label0 = decode_predictions(yhat,1000)
label1 = label0[0][0]
print("The image isssss: %s  %.4f%%" % (label1[1], label1[2]))
print(label0)
print(label1[1], label1[2])


# yhat = kclassifier.predict(img)
top_ten_indexes = list(yhat[0].argsort()[-10:][::-1])
top_probs = yhat[0, top_ten_indexes]
# labels = [label_to_name(i) for i in top_ten_indexes]
print(top_probs)
print(top_ten_indexes)
