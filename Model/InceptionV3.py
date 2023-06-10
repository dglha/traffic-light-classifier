from keras import Sequential
from keras.applications import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization


class InceptionV3Custom:
    @staticmethod
    def build(n_classes, freeze_layers = True):
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        top_model = Sequential()
        top_model.add(base_model)
        top_model.add(GlobalAveragePooling2D())
        top_model.add(Dropout(0.5))
        top_model.add(Dense(1024, activation='relu'))
        top_model.add(BatchNormalization())
        top_model.add(Dropout(0.5))
        top_model.add(Dense(512, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(128, activation='relu'))
        top_model.add(Dense(n_classes, activation='softmax'))

        if freeze_layers:
            for layer in base_model.layers:
                layer.trainable = False

        return top_model
