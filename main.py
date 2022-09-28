import cv2
import tensorflow as tf
from PIL import Image
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, CarliniL2Method, CarliniLInfMethod, \
    ProjectedGradientDescent, DeepFool, ThresholdAttack, PixelAttack, SpatialTransformation, SquareAttack, ZooAttack, \
    BoundaryAttack, HopSkipJump, SaliencyMapMethod
from art.estimators.classification import KerasClassifier
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
import numpy as np
# Disabling eager execution from TF 2
tf.compat.v1.disable_eager_execution()

model = VGG16(weights='imagenet')


kclassifier = KerasClassifier(model = model, clip_values=(0,1), use_logits=False)

attack = FastGradientMethod(estimator=kclassifier, eps=0.2)


img = load_img("canoe1.JPEG",  target_size=(224, 224))

img = img_to_array(img)
print(img)
img = img.reshape((1, 224, 224, 3))


ct = np.array([421])
adv = attack.generate(img, ct)
np.save("advers.npy", adv)

adv = adv.reshape((224, 224, 3))
Img = Image.fromarray((adv).astype(np.uint8))
filename = "%s.png" % ('advers')
Img.save(filename)