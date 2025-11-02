# model.py - Example training script
# Use this to train a real model on FER2013 or your own dataset.
# The app will work with a Keras .h5 model saved as 'emotion_model.h5' in the project root.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def make_model():
    model = Sequential([
        Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1)),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Example usage:
# model = make_model()
# model.fit(train_generator, epochs=30, validation_data=val_generator)
# model.save('emotion_model.h5')
