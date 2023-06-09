import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import pandas as pd

data_dir = r'C:/Users/User/Desktop/orl_faces'
class_names = ['elifkusumbaltanem', 'melis', 'enes']
num_classes = len(class_names)
img_size = (64, 64)

# Veri setini ve etiketleri depolamak için boş listeler oluşturun
images = []
labels = []

# Veri setindeki her sınıf için döngü
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    label = class_names.index(class_name)

    # Sınıf klasöründeki her görüntü için döngü
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)

        # Görüntüyü ve etiketi listelere ekleyin
        images.append(img)
        labels.append(label)

# Veri setini NumPy dizilerine dönüştürün
images = np.array(images)
labels = np.array(labels)

# Veri setini eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalizasyon (0-255 aralığını 0-1 aralığına dönüştürme)
#HOG yerine kullanıldı
X_train_normalized = normalize(X_train.reshape(X_train.shape[0], -1))
X_test_normalized = normalize(X_test.reshape(X_test.shape[0], -1))

# Yapay sinir ağı modelini oluşturun
model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(124, activation='relu'),
    tf.keras.layers.Dropout(0.01),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.01),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Modeli derleyin
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitin
model.fit(X_train_normalized, y_train, epochs=30, batch_size=128, validation_data=(X_test_normalized, y_test))

# Modelin performansını değerlendirin
test_loss, test_accuracy = model.evaluate(X_test_normalized, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Örnek bir görüntüyü yükleme ve ön işleme
example = cv2.imread(r'C:\Users\User\Desktop\image\elifkusumbaltanesi.jpeg', cv2.IMREAD_GRAYSCALE)
#example = cv2.imread(r'C:\Users\User\Desktop\image\melis.jpeg', cv2.IMREAD_GRAYSCALE)
#example = cv2.imread(r'C:\Users\User\Desktop\image\enes.jpeg', cv2.IMREAD_GRAYSCALE)
example = cv2.resize(example, img_size)

example = example.reshape(1, -1)

#query_image = query_image / 255.0
example = normalize(example)

example = np.expand_dims(example, axis=0)

# Tahmin yapma
predictions = model.predict(example)
predicted_label = class_names[np.argmax(predictions)]

print("predict class:", predicted_label)
# excel yazdırma
data = {'Name': [predicted_label], 'Label': ['+']}
df = pd.DataFrame(data)
df.to_excel('attendance.xlsx', index=False)
print("yoklama alındı ")
