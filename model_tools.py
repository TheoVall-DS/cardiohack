from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from keras.models import Sequential
from tensorflow_addons.metrics import F1Score

class CNNModel:
    """Create convolutional neural network."""
    
    def __init__(self):
        """Initialize self."""
        pass
    
    def create_and_compile(self):
        """Create and compile Keras model."""
        model = Sequential()
        model.add(Conv1D(filters=16, kernel_size=10, activation='relu'))
        model.add(MaxPooling1D(pool_size=5))
        model.add(Conv1D(filters=32, kernel_size=10, activation='relu'))
        model.add(MaxPooling1D(pool_size=10))
        model.add(Conv1D(filters=64, kernel_size=10, activation='relu'))
        model.add(MaxPooling1D(pool_size=15))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=2, activation='softmax'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[F1Score(num_classes=2)])
        
        return model