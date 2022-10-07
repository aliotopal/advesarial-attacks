
from attacks import *
# models: ['VGG16','VGG19','ResNet50','ResNet101','ResNet152','DenseNet121','DenseNet169','DenseNet201','MobileNet','MNASNet']
# attacks: ['FGSM','BIM','PGDL1','PGDL2','PGDInf','CWi','DeepFool']

## Set-up ATTACK:
model = 'VGG16'
attack = 'DeepFool'
targeted = True
ct = np.array([306])  # target category [0, 1000) if targeted, else enter any number

## load a clean image
ancestorImg = load_img("acorn1.JPEG",  target_size=(224, 224)) # original any-size image, .jpeg, .jpg, .png

## Creating ATTACK:
attack1 = genAttack(ancestorImg, attack, model,targeted, ct)
adversImg = attack1.getAdversImg()

## Report:
idx, labelValue = attack1.predictAdv(adversImg)


