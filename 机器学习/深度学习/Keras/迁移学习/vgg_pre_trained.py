## 该脚本实现：使用在ImageNet上训练好的VGG16作为预训练网络进行fine-tune，实现cifar10的分类
from keras import Sequential
from keras.applications import VGG16
from keras.layers import Dense,Conv2D,ZeroPadding2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD
from cifar_loader import *
from sklearn.metrics import log_loss,accuracy_score

def vgg16_model(img_row, img_col, channel, num_classes):
    model = Sequential()
    # model.add(ZeroPadding2D(1, 1), input_shape = (img_row, img_col, channel))
    model.add(Conv2D(64, (3, 3), padding='same',input_shape=(img_row,img_col,channel), activation='relu')) #224
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D((2,2), strides=(2,2))) #112 注意步长是2才能下采样！
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2))) #56
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2))) #28
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2))) #14
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2))) #7


    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    ## 加载预训练模型在ImageNet上训练好的权重
    model.load_weights('C:\\Users\\chiyuan\\.keras\\models\\vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    ## 去掉最后一层FC，用自己的代替
    model.layers.pop() #去掉最后一层
    model.outputs = [model.layers[-1].output]  #加中括号将多个结果直接放进一个列表
    model.layers[-1].outbound_nodes = []  ##???????????????
    model.add(Dense(num_classes, activation='softmax'))

    ## 保持前十层权重不变（freeze）
    for layer in model.layers[:10]:
        layer.trainable = False

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    img_rows, img_cols = 224, 224
    channel = 3
    num_classes = 10
    batch_size = 16
    epoch = 10
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_cifar10_keras(img_rows, img_cols)
    model = vgg16_model(img_rows, img_cols, channel, 10)
    model.fit(X_train, Y_train, shuffle=True,
              epochs=epoch, batch_size=batch_size,
              validation_data=(X_val, Y_val),
              verbose=1)
    prediction_test = model.predict(X_test, batch_size = batch_size, verbose=1)
    accuracy = accuracy_score(Y_test, prediction_test)














