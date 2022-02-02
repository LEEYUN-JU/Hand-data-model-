import os, glob, numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import optimizers

#GPU CPU 연산 
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True #allow_growth 옵션: 초기에 메모리를 할당하지 않고 세션을 시작한 후에 더 많은 GPU 메모리가
#필요할때 메모리 영역을 확장한다.
# ex) config.gpu_options.per_process_gpu_memory_fraction = 0.6: 전체 GPU소모량을 정하고 싶을때
session = tf.compat.v1.Session(config=config) #세션 셜정

result = []

folder = ["All"]
# folder = ["Finger", "Hand", "Angle", "Effort", "All"]
normalization = ["Normalization"]
# normalization = ["Standard", "Minmax", "Robust", "Normalization"]

defalut = "D:/Classification/Barrett_Hand_codes"

for f, number in enumerate(range(len(folder))):    
    for i, title in enumerate(range(len(normalization))):
        X_train, X_test, y_train, y_test = np.load(defalut + '/dataset/dataset_%s_%s.npy'%(folder[number], normalization[title])) #데이터 셋 로딩

        categories = ["ball", "banana", "can","cube","pear","spam","strawberry","tennis"]
        nb_classes = len(categories)
    
        X_train = X_train.astype(float) / 255 #X 데이터는 흑백으로 구성되어 있음. 0~255의 값을 0~1의 값으로 Nomalize
        X_test = X_test.astype(float) / 255
    
        #print(device_lib.list_local_devices())
    
        with K.tf_ops.device('/device:CPU:0'):
            model = Sequential() #선형 모델
            model.add(Conv2D(32, (3,3), padding="same", input_shape=X_train.shape[1:], activation='relu'))
            #conv2D(컨볼루션 필터 수, 컨볼루션 커널의 수, 입력형태(샘플 수 제외), 활성화 함수)
            # model.add(MaxPooling2D(pool_size=(2,2))) # 차원에 대한 Downsampling 수행
            model.add(Dropout(0.2)) #Overfitting 을 방지하기 위한 Dropout
            
            model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Dropout(0.2))
         
            # model.add(Conv2D(128, (3,3), padding="same", activation='relu'))
            model.add(Conv2D(128, (3,3), padding="same", activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Dropout(0.2))
        
            # model.add(Conv2D(256, (3,3), padding="same", activation='relu'))
            # model.add(MaxPooling2D(pool_size=(2,2)))
            # model.add(Dropout(0.4))
        
            model.add(Flatten()) 
            
            model.add(Dense(256, activation='relu')) #은닉 계층
            model.add(Dropout(0.4))
            model.add(Dense(nb_classes, activation='softmax')) 
            
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
            model_path = defalut + '/models/model_%s_%s.model'%(folder[number], normalization[title])
            checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
            #verbose : 해당 함수의 진행 사항의 출력 여부
            #save_best_only : 모델의 정확도가 최고값을 갱신했을때만 저장
            early_stopping = EarlyStopping(monitor='val_loss', patience=10)
            #최고 성능의 모델이 찾아졌을 경우 학습 중단. ex: epochs = 100 pa    tience = 10, 100번 학습할동안 6번 이내로 찾아내지 못하면 중단
    
        model.summary()
    
        #모델 학습시키기
        history = model.fit(X_train, y_train, batch_size=64, epochs=10, 
                            #validation_data=(X_test, y_test), callbacks=[checkpoint])
                            validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping])
    
        result.append(folder[number] + "_" + normalization[title] + " : " + "%.4f"% (model.evaluate(X_test, y_test)[1]))
        #print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))#모델 평가하기
    
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        epochs = range(1, len(acc) + 1)
        
        plt.plot(epochs, loss, 'b', label = 'Training loss')
        plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
        
        plt.plot(epochs, acc, 'y', label = 'T Acc')
        plt.plot(epochs, val_acc, 'g', label = 'V acc')
        
        plt.title(folder[number] + "_" + normalization[title] + " : " + "%.4f"% (model.evaluate(X_test, y_test)[1]))
        plt.legend(loc = 'best')
        plt.figure(figsize=(16,10))
        plt.show()
        
for pp in range(0, 4):
    print(result[pp])