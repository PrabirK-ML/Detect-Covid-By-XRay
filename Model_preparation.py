from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint


class ModelPreparation:

    def create_model_structure(self,classes):
        model=Sequential()
        model.add(Conv2D(32,(3,3),input_shape=(299,299,3)))
        model.add(Conv2D(64,(3,3),activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(256,(3,3),activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(128,(3,3),activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(128,(3,3),activation='relu'))
        model.add(Flatten())
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes,activation='softmax'))

        model.summary()
        plot_model(model,to_file='model.png',show_shapes=True)
        return model

    def train_model(self,model,traindatapath,valdatapath):
        
        datagen=ImageDataGenerator(rescale=1/255)
        traingenerator=datagen.flow_from_directory(traindatapath,
                                                   target_size=(299,299),
                                                   batch_size=20,
                                                   class_mode='categorical'
                                                   )
        valgenerator=datagen.flow_from_directory(valdatapath,
                                                target_size=(299,299),
                                                batch_size=20,
                                                class_mode='categorical'
                                                )
        adam=Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
        mc=ModelCheckpoint('Best_model.h5',monitor='val_accuracy',verbose=1,save_best_only=True)
        rl=ReduceLROnPlateau(monitor='val_accuracy',factor=0.1,patience=3,cool_down=1,verbose=1)
        
        model.fit(traingenerator,steps_per_epoch=2000,epochs=10,validation_data=valgenerator,validation_steps=40,verbose=1,callbacks=[mc,rl])

        return None
        
        
        
        



if __name__=="__main__":
    mp=ModelPreparation()
    model=mp.create_model_structure(4)
    mp.train_model(model,'Data/train','Data/val')
        
        
