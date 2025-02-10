import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
import string


def load_data(file_path):
    
    data = pd.read_csv(file_path)
    
    
    X = data.iloc[:, 1:].values  
    y = data.iloc[:, 0].values   
    
    
    y = np.array([string.ascii_uppercase[i] for i in y])
    
    
    X = X.reshape(-1, 28, 28, 1)
    
    
    X = X / 255.0
    
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_onehot = to_categorical(y_encoded)
    
    
    joblib.dump(label_encoder, 'label_encoder.joblib')
    
    return X, y_onehot


def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    
    X, y = load_data('handwritten_data_785.csv')
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    model = create_model(num_classes=26)  
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    
    model.save('letter_recognition_model.h5')
    
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'\nTest accuracy: {test_acc:.3f}') 