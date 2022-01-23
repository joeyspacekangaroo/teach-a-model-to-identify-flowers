#WARNING: I could not figure out how to get past step 3 on this project

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
%matplotlib inline

dataset, dataset_info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)
dataset_size = dataset_info.splits['train'].num_examples

test_set, training_set, validation_set = dataset['test'], dataset['train'], dataset['validation']

def preprocessWithAspectRatio(image,label):
    resized_image=tf.image.resize_with_pad(image,299,299)
    final_image=keras.applications.xception.preprocess_input(resized_image)
    return final_image,label
	
trainPipe=training_set.map(preprocessWithAspectRatio,num_parallel_calls=32).batch(128)
testPipe=test_set.map(preprocessWithAspectRatio,num_parallel_calls=32).batch(128)
valPipe=validation_set.map(preprocessWithAspectRatio,num_parallel_calls=32).batch(128)

basemodel = keras.applications.xception.Xception(weights='imagenet',include_top=False)

avg = keras.layers.GlobalAveragePooling2D()(basemodel.output)
output = keras.layers.Dense(102,activation="softmax")(avg)
model = keras.models.Model(inputs=basemodel.input,outputs=output)
model.summary()

for layers in basemodel.layers:
    layers.trainable = False

for layer in model.layers:
    print(layer.trainable)
	
#inputting into the stack

checkpoint_callback  = keras.callbacks.ModelCheckpoint('flowersModel.h5',save_best_only=True)

earlystop_callback = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights=True)

ss = 5e-1

optimizer = keras.optimizers.SGD(learning_rate=ss)

model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,metrics =["accuracy"])

model.fit(testPipe,validation_data=valPipe,epochs=25,callbacks=[checkpoint_callback,earlystop_callback])

model.summary()

#could not figure out how to get vectors out of the dataset

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(n_components = 2)

X2D = pca.fit_transform(model)




