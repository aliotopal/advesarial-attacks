
import tensorflow as tf
from PIL import Image
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, CarliniL2Method, CarliniLInfMethod, \
    ProjectedGradientDescent, DeepFool, ThresholdAttack, PixelAttack, SpatialTransformation, SquareAttack, ZooAttack, \
    BoundaryAttack, HopSkipJump, SaliencyMapMethod
from art.estimators.classification import KerasClassifier, TensorFlowV2Classifier
# import imagenet_stubs
# from imagenet_stubs.imagenet_2012_labels import label_to_name, name_to_label

from keras_preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16

import numpy as np
# Disabling eager execution from TF 2
# tf.compat.v1.disable_eager_execution()
model = VGG16(include_top=True, weights='imagenet', classifier_activation='softmax')
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
# model1 = VGG16(weights='imagenet')
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# mean = [0.485*255, 0.456*255, 0.406*255]
# std = [0.229*255, 0.224*255, 0.225*255]



kclassifier = TensorFlowV2Classifier(model = model,
                                     nb_classes=1000,
                                     loss_object=loss,
                                     input_shape=(224,224,3),
                                     clip_values=(0.0,1.0),
                                     preprocessing=(mean,std)
                                     )


attackPGD = ProjectedGradientDescent(estimator=kclassifier,
                                  eps=8.0/255.0,
                                  max_iter=32,
                                  eps_step=2.0/255.0,
                                  targeted=True)

attackFGSM = FastGradientMethod(estimator=kclassifier,
                                  eps=2.0/255.0,
                                  eps_step=0.01,
                                  targeted=True)


img = load_img("acorn1.JPEG",  target_size=(224, 224))
img = img_to_array(img)
img = (np.expand_dims(img, axis=0)/255.0).astype(np.float32)


# img = img.reshape((1, 224, 224, 3))
# predictions = kclassifier.predict(img)
# print(predictions)

ct = np.array([309])
adv = attackPGD.generate(img, ct)
y_pred_adv = kclassifier.predict(adv)
top_ten_indexes = list(y_pred_adv[0].argsort()[-10:][::-1])
top_probs = y_pred_adv[0, top_ten_indexes]
# labels = [label_to_name(i) for i in top_ten_indexes]
print(top_probs)
print(top_ten_indexes)




adv = adv*255

np.save("adversPGD.npy", adv)

adv = adv.reshape((224, 224, 3))
Img = Image.fromarray((adv).astype(np.uint8))
filename = "%s.png" % ('adversPGD')
Img.save(filename)





