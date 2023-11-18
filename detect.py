from keras.models import load_model
import cv2  
import numpy as np
import cvzone

np.set_printoptions(suppress=True)

model = load_model("models/keras_model.h5", compile=False)

class_names = ['bom estado', 'com defeito']

img = cv2.imread('test/com defeito/1.png')
img = cv2.resize(img, (800, 800))

image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

image = (image / 127.5) - 1

prediction = model.predict(image)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

texto1 = f"Classificacao: {class_name}"
texto2 = f"Taxa de acerto: {str(np.round(confidence_score * 100))[:-2]} %"

print(texto1,texto2)

cvzone.putTextRect(img, texto1,(50,50))
cvzone.putTextRect(img, texto2,(50,100))

cv2.imshow('Detector de Pneus Defeituosos', img)
cv2.waitKey(0)