import csv
import cv2
import numpy as np
import sklearn

DATA_FOLDER='/home/user/data/datasets/DrivingSIM/mouse/'
CSV_FILE='driving_log.csv'
CSV_PATH=DATA_FOLDER+CSV_FILE

# load data
lines = []
with open(CSV_PATH) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print(len(train_samples))
print(len(validation_samples))

train_size = len(train_samples)*3*2
val_size = len(validation_samples)*3*2

import time

correction=[0,0.2,-0.2]
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            start_time=time.time()
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                #name = './IMG/'+batch_sample[0].split('/')[-1]
                for i in range(3):
                    name = batch_sample[i]
                    image = cv2.imread(name)
                    image = cv2.resize(image,(160,80))
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    angle = float(batch_sample[3])
                    images.append(image)
                    angles.append(angle+correction[i])

            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, angles):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)
            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            #print("--- %s seconds ---" % (time.time() - start_time))
            yield sklearn.utils.shuffle(X_train, y_train)

#images = []
#measurements = []
## corrction factor for center, left, right images
#
#for line in lines:
#    for i in range(3):
#        img_path=line[0]
#        image = cv2.imread(img_path)
#        image = cv2.resize(image,(160,80))
#        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#        images.append(image)
#        measurement = float(line[3]) + correction[i]
#        measurements.append(measurement)
#
#augmented_images, augmented_measurements = [], []
#for image, measurement in zip(images, measurements):
#    augmented_images.append(image)
#    augmented_measurements.append(measurement)
#    augmented_images.append(cv2.flip(image,1))
#    augmented_measurements.append(measurement*-1.0)
#
#X_train = np.array(augmented_images)
#y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Reshape, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint


BATCH_SIZE=32
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

# training model
model = Sequential()
#model.add(Cropping2D(cropping=((25,9),(0,0)),input_shape=(80,160,3)))
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(80,160,3)))
#model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Cropping2D(cropping=((35,13),(0,0)),data_format='channels_first'))
model.add(Conv2D(filters=24, kernel_size=(5,5),subsample=(2,2), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=36, kernel_size=(5,5),subsample=(2,2), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=48, kernel_size=(5,5),subsample=(2,2), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3,3),subsample=(1,1), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3,3),subsample=(1,1), activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

filepath="weights-improvement-{epoch:02d}.hdf5"
checkpointer = ModelCheckpoint(filepath, verbose=1, save_best_only=True)

model.fit_generator(train_generator, steps_per_epoch= \
            train_size/BATCH_SIZE, validation_data=validation_generator, \
                    nb_val_samples=val_size/BATCH_SIZE, nb_epoch=3,callbacks=[checkpointer])
#model.fit(X_train,y_train,validation_split=0.2,shuffle=True, nb_epoch=3)

model.save('model.h5')
