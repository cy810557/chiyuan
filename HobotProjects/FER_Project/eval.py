import mxnet as mx
import cv2
from collections import namedtuple
from os.path import join
from mxnet import nd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import sys
##from dataset_loader import DatasetLoader

def format_image(image):
    '''Detecte face (using OpenCV cascade classifier) from given frame, and format to Mxnet input format'''
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    # None is we don't found an image
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    # Chop image to face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    # Resize image to network size
    try:
        image = cv2.resize(image, (input_height, input_width),
                           interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None
    # cv2.imshow("Lol", image)
    # cv2.waitKey(0)
    image = nd.array(image).expand_dims(-1).transpose((2, 0, 1)).expand_dims(0)
    return image

def load_images(img):
    '''Duplicated'''
    if type(img)==str:
        frame = cv2.imread(img)
    else:  #video frame
        frame = img
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(input_height, input_width),interpolation=cv2.INTER_CUBIC)
    img = nd.array(img).expand_dims(-1).transpose((2,0,1)).expand_dims(0)
    return img, frame 
def draw_bars(frame, result, feelings_faces):
    '''Draw visulization bars'''
    if result is None:
        print('prob shuold not bereturnne Type. Please check...')
        return
    for index, emotion in enumerate(EMOTIONS):
        cv2.putText(frame, emotion, (10, index * 22 + 20),
                            cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
        cv2.rectangle(frame,  (130, index * 22 + 10),
                      (130 + int(result[0][index] * 100),
                     (index + 1) * 22 + 4), (255, 0, 0), -1)
        face_image = feelings_faces[np.argmax(result[0])]
        # decoration
    for c in range(0, 3):
        frame[EMOJI_H:EMOJI_H+EMOJI_SIZE, EMOJI_R:EMOJI_R+EMOJI_SIZE, c] = \
        face_image[:, :, c] * (face_image[:, :, 3] / 255.0) + \
        frame[EMOJI_H:EMOJI_H+EMOJI_SIZE, EMOJI_R:EMOJI_R+EMOJI_SIZE, c] * (1.0 - face_image[:, :, 3] / 255.0)
    if len(sys.argv)>1:
        plt.imshow(frame[...,::-1])
        plt.show()
    else:
        cv2.imshow('FER', frame)




CASC_PATH = 'CASC/haarcascade_frontalface_default.xml'
EMOJI_H, EMOJI_R = 160, 10
EMOJI_SIZE = 80
model_prefix = 'models/My_FER_model'
num_epoch = 10000
input_width, input_height = 48, 48

cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

EMOTIONS = ['angry', 'disgusted', 'fearful',
            'happy', 'sad', 'surprised', 'neutral']
			
def get_emojis(EMOTIONS):
    feelings_faces = []
    for index, emotion in enumerate(EMOTIONS):
        feelings_faces.append(cv2.resize(cv2.imread('../emojis/' + emotion + '.png', -1), (80, 80)))
    return feelings_faces
def build_model(model_prefix, num_epoch):
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, num_epoch)
    data = mx.sym.Variable(name='data')
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(0), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data',(1,1,input_height,input_width))])
    mod.set_params(arg_params, aux_params, allow_missing=True)
    Batch = namedtuple('Batch', ['data'])
    return mod, Batch

if len(sys.argv)>1:
    mod, Batch = build_model(model_prefix, num_epoch)
    frame = cv2.imread(sys.argv[1])
    img = format_image(frame)
    mod.forward(Batch([nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    print('predictions: {0}'.format(list(zip(EMOTIONS,list(prob[0])))))
    feelings_faces = get_emojis(EMOTIONS)
    draw_bars(frame, prob, feelings_faces)