## 该脚本尝试使用keras库中不同的预训练模型进行图像分类
## 注意：该脚本尝试定义一些接口以供用户输入参数。
from keras.applications import ResNet50, InceptionV3, VGG16
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img #同PIL.Image.open,返回一个PIL类型数据
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
## Step1: 输入参数construct
ap = argparse.ArgumentParser()
ap.add_argument('-i','--image',required = True, help = 'path to the input image')
# '-i'是用户传入时用的， '--image'是调用时（如args.image）用的
ap.add_argument('-model', '--model', type = str, default='vgg16',
                help = 'name of pre-trained network to use')
args = vars(ap.parse_args())
# print(args['image'])
# print(args['model'])
# plt.imshow(load_img(args['image']))
# plt.show()

## Step2: 定义一个字典接收用户输入
MODELS = {'vgg16':VGG16,
          'inception':InceptionV3,
          'resnet':ResNet50}
if args['model'] not in MODELS.keys():
    raise AssertionError('--model command line should be a key in [vgg16,inception and resnet]!')

## Step3： 加载预定义模型的imagenet权重
# 注意：VGG16，VGG19和ResNet均接受224×224输入图像，而Inception V3和Xception需要299×299像素输入
if args['model'] == 'inception':
    input_shape = (299, 299)
    preprocess_func = preprocess_input #inception_v3的预处理函数，注意只取函数名，不加括号（不调用）
else:
    input_shape = (224, 224)
    preprocess_func = imagenet_utils.preprocess_input #其他模型通用的预处理函数
    #注意：keras中的preprocess_input()函数输入可以是numpy或者tensor，可以是3D或4D，结果会经过zero-center
    #     若输入tensor则输出[-1:1]， numpy则输出


print('[INFO] loading {}...'.format(args['model']))
Network = MODELS[args['model']]
model = Network(weights='imagenet')

## Step4: load并预处理自己的图像
print('[INFO] loading and pre-processing image')
img = load_img(args['image'], target_size=input_shape)
plt.subplot(211)
plt.imshow(img)
plt.axis('off')

img = img_to_array(img)  #PIL-> numpy
# 注意：此时的img尺寸为：[inputshape[0], inputshape[1], 3],但是网络输入为一个4Darray，因此要将img尺寸改成[1,x,y,3]
img = np.expand_dims(img, axis=0)
img = preprocess_func(img) #使用上面选中的函数进行预处理(输入先转换成numpy)

## Step5：对自己的图像进行分类
print('[INFO] classify image with model {}'.format(args['model']))
preds = model.predict(img)
# P = imagenet_utils.decode_predictions(preds)
P = imagenet_utils.decode_predictions(preds, top=3)[0]

plt.subplot(212)
X1 = list(reversed(range(len(P))))
bar_preds = [pr[2] for pr in P]
labels = (pr[1] for pr in P)
plt.barh(X1, bar_preds, alpha=0.5)

plt.yticks(X1, labels)
plt.xlabel('Probability')
plt.xlim(0, 1.1)
plt.tight_layout()
plt.show()
# for (i, (imagenetID, label, prob)) in enumerate(P[0]):
#     print('{}. {}: {:.2f}%'.format(i+1, label, prob*100))


