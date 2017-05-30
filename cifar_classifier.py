from keras.datasets import cifar10
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

class cifar_classifier(object):
    """A blueprint for models to classiy
       CIFAR-10 dataset
    """

    def __init__(self):
        """Initialize and load the CIFAR database and Initialize models"""
        (self.X_train, self.y_train), (self.X_test,self.y_test) = cifar10.load_data()
        self.models = self._models()
        self.trained_model = None

    def _rescale(self, X_train, X_test):
        """Rescale pixel values
           Note 255 - white
           Note 0 - black
           1-244- Some sort of grey
           with 0 to 1 - 1 is white 0 is black
        """
        return (X_train.astype('float32')/255, X_test.astype('float32')/255)

    def _one_hot_encode(self, y_train, y_test):
        """ One hot_encode_outputs"""
        return ( np_utils.to_categorical(y_train,10), np_utils.to_categorical(y_test,10))

    def _train_cnn(self):
        """Train the model using CNN architecture"""
        #rescale image pixel values
        (X_train, X_test) = self._rescale(self.X_train, self.X_test)

        #One-hot encode outputs
        (y_train, y_test) = self._one_hot_encode(self.y_train,self.y_test)
        # create and configure augmented image generator
        datagen = ImageDataGenerator(
            width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
            height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
            horizontal_flip=True) # randomly flip images horizontally

        # fit augmented image generator on data
        datagen.fit(X_train)
        #Train model
        model = self.models['cnn']
        checkpointer = ModelCheckpoint(filepath="cifar10.model.cnn.hdf5", verbose = 1,save_best_only = True)
        model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),steps_per_epoch=X_train.shape[0] // 32, epochs=51,verbose=2, validation_data=(X_test,y_test),callbacks=[checkpointer])
        # evaluate and print test accuracy
        score = model.evaluate(X_test, y_test, verbose=0)
        print('\n', 'Test accuracy:', score[1])
        return model

    def _train_mlp(self):
        """Train the model using MLP architecture"""
        #rescale image pixel values
        (X_train, X_test) = self._rescale(self.X_train, self.X_test)

        #One-hot encode outputs
        (y_train, y_test) = self._one_hot_encode(self.y_train,self.y_test)


        #Train model
        model = self.models['mlp']
        checkpointer = ModelCheckpoint(filepath="cifar10.model.mlp.hdf5", verbose = 1,save_best_only = True)
        model.fit(X_train, y_train, epochs=25,batch_size=64,verbose=2, validation_data=(X_test,y_test),callbacks=[checkpointer])
        # evaluate and print test accuracy
        score = model.evaluate(X_test, y_test, verbose=0)
        print('\n', 'Test accuracy:', score[1])
        return model

    def _models(self):
        """Here we define two different models to train for CIFAR-10
            1. CNN
            2. MLP
        """
        cnn = Sequential()
        cnn.add(Conv2D(filters=16, kernel_size=2, strides=1, padding='same', activation='relu', input_shape=(32, 32, 3)))
        cnn.add(Conv2D(filters=32, kernel_size=3    , strides=1, padding='same', activation='relu'))
        cnn.add(MaxPooling2D(pool_size=2))
        cnn.add(Conv2D(filters=64, kernel_size=4, strides=1, padding='same', activation='relu'))
        cnn.add(MaxPooling2D(pool_size=2))
        cnn.add(Conv2D(filters=128, kernel_size=4, strides=1, padding='same', activation='relu'))
        cnn.add(MaxPooling2D(pool_size=2))
        cnn.add(Dropout(0.2))
        cnn.add(Flatten())
        cnn.add(Dropout(0.2))
        cnn.add(Dense(512,activation='relu'))
        cnn.add(Dropout(0.2))
        cnn.add(Dense(10, activation='softmax'))
        #cnn.summary()

        #Compile cnn
        cnn.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        mlp = Sequential()
        mlp.add(Flatten(input_shape=self.X_train.shape[1:]))
        mlp.add(Dense(2048, activation='relu'))
        mlp.add(Dropout(0.2))
        mlp.add(Dense(1024, activation='relu'))
        mlp.add(Dropout(0.2))
        mlp.add(Dense(512,activation='relu'))
        mlp.add(Dropout(0.2))
        mlp.add(Dense(10, activation='softmax'))
        #mlp.summary()

        #Compile mlp
        mlp.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        return {'cnn' : cnn, 'mlp': mlp}

    def train(self, method):
        """Wrapper method to call the actual training method"""
        if method in self.models:
            method = ''.join(['_train_', method])
            method = getattr(self, method)
            self.trained_model = method()
        else:
            raise("Not a Available method, methods available are \n 1.cnn \n2.mlp")

    def get_model(self, method):
        """get the model of a given method"""
        if method in self.models:
            """ return  actual model"""
            return self.models[method]

    def get_test_data(self):
        """Get test data"""
        return (self.X_test, self.y_test)

def train_models():
    """Train both cnn and mlp"""
    #Pick and train CNN
    print("Training CNN\n")
    cc = cifar_classifier()
    cc.train('cnn')

    #train mlp
    print("\nTraining MLP\n")
    cc.train('mlp')

if __name__ == '__main__':
    train_models()
