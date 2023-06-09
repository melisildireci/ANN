import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten

data_dir = r'C:/Users/User/Desktop/orl_faces2'
class_names = ['furkan', 'semih', 'eymen']
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
X_train = X_train / 255.0
X_test = X_test / 255.0

# Yapay sinir ağı modelini oluşturun
model = Sequential()
model.add(Flatten(input_shape=(img_size[0], img_size[1])))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Modeli derleyin
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitin
model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_test, y_test))

# Modelin performansını değerlendirin
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Kamera üzerinden tahmin yapma
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü gri tonlamaya dönüştürün ve yeniden boyutlandırın
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, img_size)

    # Normalizasyon
    gray = gray / 255.0

    # Tahmin yapma
    prediction = model.predict(np.array([gray]))
    predicted_class = class_names[np.argmax(prediction)]

    # Tahmin sonucunu ekrana yazdırma
    cv2.putText(frame, "Predicted: {}".format(predicted_class), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow
