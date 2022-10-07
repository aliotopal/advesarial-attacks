import tensorflow as tf
import numpy as np
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, CarliniL2Method, CarliniLInfMethod, \
    ProjectedGradientDescent, DeepFool, ThresholdAttack, PixelAttack, SpatialTransformation, SquareAttack, ZooAttack, \
    BoundaryAttack, HopSkipJump, SaliencyMapMethod
from art.estimators.classification import KerasClassifier, TensorFlowV2Classifier
from keras_preprocessing.image import load_img, img_to_array


class genAttack:
    def __init__(self, image, attack, network_name, targeted, ct):
        self.ancestor = image
        self.network_name = network_name
        self.model1 = self.create_model(network_name)
        self.attack = attack
        self.targeted = targeted
        self.ct = ct


    def getAdversImg(self):
        '''image should be in .png, .jpeg, .jpg format
        ct: index number of target label in numpy array format eg. np.array([223])
        RETURNS: adversarial image in .npy format, shape: (1, 224, 224, 3)'''
        # model = self.create_model(self.model1)
        img = img_to_array(self.ancestor)
        img = (np.expand_dims(img, axis=0) / 255.0).astype(np.float32)
        img = img.reshape((1, 224, 224, 3))
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        kclassifier = TensorFlowV2Classifier(model = self.model1 ,
                                             nb_classes=1000,
                                             loss_object=loss,
                                             input_shape=(224,224,3),
                                             clip_values=(0.0,1.0),
                                             preprocessing=(0.5,0.5))
        if self.attack == "PGDInf":
            attackPGDL1 = ProjectedGradientDescent(estimator=kclassifier,
                                              eps=8.0/255.0,
                                              norm=np.inf,
                                              max_iter=100,
                                              eps_step=2.0/255.0,
                                              targeted=True,
                                              verbose=True)
            if self.targeted:
                advers = attackPGDL1.generate(x=img, y=self.ct) * 255
            else:
                advers = attackPGDL1.generate(x=img) * 255
            return advers


        elif self.attack == "FGSM":
            attackFGSM = FastGradientMethod(estimator=kclassifier,
                                              eps=2.0/255.0,
                                              eps_step=0.01,
                                              targeted=True)
            if self.targeted:
                advers = attackFGSM.generate(x=img, y=self.ct) * 255
            else:
                advers = attackFGSM.generate(x=img) * 255
            return advers

        elif self.attack == "DeepFool":
            attackDeepFool = DeepFool(classifier=kclassifier,
                                      # max_iter=100,
                                      # epsilon=0.01,
                                      # nb_grads=10,
                                      # batch_size=1,
                                      verbose=True)
            advers = attackDeepFool.generate(x=img) * 255
            return advers

    def create_model(self, network_name):

        if network_name == 'VGG16':
            from tensorflow.keras.applications.vgg16 import VGG16
            model = VGG16(weights='imagenet')

        elif network_name == 'VGG19':
            from tensorflow.keras.applications.vgg19 import VGG19
            model = VGG19(weights='imagenet')

        elif network_name == 'ResNet50':
            from tensorflow.keras.applications.resnet import ResNet50
            model = ResNet50(weights='imagenet')

        elif network_name == 'ResNet101':
            from tensorflow.keras.applications.resnet import ResNet101
            model = ResNet101(weights='imagenet')

        elif network_name == 'ResNet152':
            from tensorflow.keras.applications.resnet import ResNet152
            model = ResNet152(weights='imagenet')

        elif network_name == 'DenseNet121':
            from tensorflow.keras.applications.densenet import DenseNet121
            model = DenseNet121(weights='imagenet')

        elif network_name == 'DenseNet169':
            from tensorflow.keras.applications.densenet import DenseNet169
            model = DenseNet169(weights='imagenet')

        elif network_name == 'DenseNet201':
            from tensorflow.keras.applications.densenet import DenseNet201
            model = DenseNet201(weights='imagenet')

        elif network_name == 'MobileNet':
            from tensorflow.keras.applications.mobilenet_v2 import MobileNet
            model = MobileNet(weights='imagenet')

        elif network_name == 'MNASNet':
            from tensorflow.keras.applications.nasnet import NASNetMobile
            model = NASNetMobile(weights='imagenet')

        elif network_name == 'InceptionV3':
            from tensorflow.keras.applications.inception_v3 import InceptionV3
            model = InceptionV3(weights='imagenet')
        return model


    def predictAdv(self, adversImg):

        if self.network_name == 'VGG16':
            from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
            image = adversImg.reshape((1, 224, 224, 3))
            yhat = self.model1.predict(preprocess_input(image))
            label0 = decode_predictions(yhat, 1000)
            idx = np.argmax(yhat)
            label1 = label0[0][0]
            print(label0)
            print("The image is: %s  %.4f%%" % (label1[1], label1[2]))
            print("Advers image index: ", idx)
            return idx, label1[2]


        elif self.network_name == 'VGG19':
            from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
            image = adversImg.reshape((1, 224, 224, 3))
            yhat = self.model1.predict(preprocess_input(image))
            label0 = decode_predictions(yhat, 1000)
            idx = np.argmax(yhat)
            label1 = label0[0][0]
            print(label0)
            print("The image is: %s  %.4f%%" % (label1[1], label1[2]))
            print("Advers image index: ", idx)
            return idx, label1[2]

        elif self.network_name == 'ResNet50':
            from tensorflow.keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
            image = adversImg.reshape((1, 224, 224, 3))
            yhat = self.model1.predict(preprocess_input(image))
            label0 = decode_predictions(yhat, 1000)
            idx = np.argmax(yhat)
            label1 = label0[0][0]
            print(label0)
            print("The image is: %s  %.4f%%" % (label1[1], label1[2]))
            print("Advers image index: ", idx)
            return idx, label1[2]

        elif self.network_name == 'ResNet101':
            from tensorflow.keras.applications.resnet import ResNet101, preprocess_input, decode_predictions
            image = adversImg.reshape((1, 224, 224, 3))
            yhat = self.model1.predict(preprocess_input(image))
            label0 = decode_predictions(yhat, 1000)
            idx = np.argmax(yhat)
            label1 = label0[0][0]
            print(label0)
            print("The image is: %s  %.4f%%" % (label1[1], label1[2]))
            print("Advers image index: ", idx)
            return idx, label1[2]

        elif self.network_name == 'ResNet152':
            from tensorflow.keras.applications.resnet import ResNet152, preprocess_input, decode_predictions
            image = adversImg.reshape((1, 224, 224, 3))
            yhat = self.model1.predict(preprocess_input(image))
            label0 = decode_predictions(yhat, 1000)
            idx = np.argmax(yhat)
            label1 = label0[0][0]
            print(label0)
            print("The image is: %s  %.4f%%" % (label1[1], label1[2]))
            print("Advers image index: ", idx)
            return idx, label1[2]

        elif self.network_name == 'DenseNet121':
            from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
            image = adversImg.reshape((1, 224, 224, 3))
            yhat = self.model1.predict(preprocess_input(image))
            label0 = decode_predictions(yhat, 1000)
            idx = np.argmax(yhat)
            label1 = label0[0][0]
            print(label0)
            print("The image is: %s  %.4f%%" % (label1[1], label1[2]))
            print("Advers image index: ", idx)
            return idx, label1[2]

        elif self.network_name == 'DenseNet169':
            from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input, decode_predictions
            image = adversImg.reshape((1, 224, 224, 3))
            yhat = self.model1.predict(preprocess_input(image))
            label0 = decode_predictions(yhat, 1000)
            idx = np.argmax(yhat)
            label1 = label0[0][0]
            print(label0)
            print("The image is: %s  %.4f%%" % (label1[1], label1[2]))
            print("Advers image index: ", idx)
            return idx, label1[2]

        elif self.network_name == 'DenseNet201':
            from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input, decode_predictions
            image = adversImg.reshape((1, 224, 224, 3))
            yhat = self.model1.predict(preprocess_input(image))
            label0 = decode_predictions(yhat, 1000)
            idx = np.argmax(yhat)
            label1 = label0[0][0]
            print(label0)
            print("The image is: %s  %.4f%%" % (label1[1], label1[2]))
            print("Advers image index: ", idx)
            return idx, label1[2]

        elif self.network_name == 'MobileNet':
            from tensorflow.keras.applications.mobilenet_v2 import MobileNet, preprocess_input, decode_predictions
            image = adversImg.reshape((1, 224, 224, 3))
            yhat = self.model1.predict(preprocess_input(image))
            label0 = decode_predictions(yhat, 1000)
            idx = np.argmax(yhat)
            label1 = label0[0][0]
            print(label0)
            print("The image is: %s  %.4f%%" % (label1[1], label1[2]))
            print("Advers image index: ", idx)
            return idx, label1[2]

        elif self.network_name == 'MNASNet':
            from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input, decode_predictions
            image = adversImg.reshape((1, 224, 224, 3))
            yhat = self.model1.predict(preprocess_input(image))
            label0 = decode_predictions(yhat, 1000)
            idx = np.argmax(yhat)
            label1 = label0[0][0]
            print(label0)
            print("The image is: %s  %.4f%%" % (label1[1], label1[2]))
            print("Advers image index: ", idx)
            return idx, label1[2]

        elif self.network_name == 'InceptionV3':
            from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
            image = adversImg.reshape((1, 224, 224, 3))
            yhat = self.model1.predict(preprocess_input(image))
            label0 = decode_predictions(yhat, 1000)
            idx = np.argmax(yhat)
            label1 = label0[0][0]
            print(label0)
            print("The image is: %s  %.4f%%" % (label1[1], label1[2]))
            print("Advers image index: ", idx)
            return idx, label1[2]
