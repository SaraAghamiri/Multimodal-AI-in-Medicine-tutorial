import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


clinical_data = pd.read_csv('clinical_data.csv')
images = np.load('echo_images.npy')
labels = pd.read_csv('labels.csv')


X_clinical_train, X_clinical_test, X_image_train, X_image_test, y_train, y_test = train_test_split(
    clinical_data, images, labels, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_clinical_train = scaler.fit_transform(X_clinical_train)
X_clinical_test = scaler.transform(X_clinical_test)
