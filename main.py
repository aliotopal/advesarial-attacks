
import tensorflow as tf
from PIL import Image
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, CarliniL2Method, CarliniLInfMethod, \
    ProjectedGradientDescent, DeepFool, ThresholdAttack, PixelAttack, SpatialTransformation, SquareAttack, ZooAttack, \
    BoundaryAttack, HopSkipJump, SaliencyMapMethod
from art.estimators.classification import KerasClassifier, TensorFlowV2Classifier

from keras_preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16

import numpy as np
# Disabling eager execution from TF 2
# tf.compat.v1.disable_eager_execution()

model1 = VGG16(weights='imagenet')

loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

kclassifier = TensorFlowV2Classifier(model = model1,
                                     nb_classes=1000,
                                     loss_object=loss,
                                     input_shape=(224,224,3),
                                     clip_values=(0.0,1.0),
                                     preprocessing=(0.5,0.5))

attackPGD = ProjectedGradientDescent(estimator=kclassifier,
                                  eps=8.0/255.0,
                                  max_iter=100,
                                  eps_step=2.0/255.0,
                                  targeted=True)

attackFGSM = FastGradientMethod(estimator=kclassifier,
                                  eps=2.0/255.0,
                                  eps_step=0.01,
                                  targeted=True)


img = load_img("acorn1.JPEG",  target_size=(224, 224))
img = img_to_array(img)
img = (np.expand_dims(img, axis=0)/255.0).astype(np.float32)


img = img.reshape((1, 224, 224, 3))
# predictions = kclassifier.predict(img)
# print(predictions)

ct = np.array([306])
adv = attackPGD.generate(img, ct)*255
np.save("adversPGD.npy", adv)

adv = adv.reshape((224, 224, 3))
Img = Image.fromarray((adv).astype(np.uint8))
filename = "%s.png" % ('adversPGD')
Img.save(filename)


