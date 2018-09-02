import keras
import numpy as np
from keras.models import load_model
import cv2
import numpy as np

test_name = 'cnn2'
model = load_model('%s.hdf5' % test_name)

cap = cv2.VideoCapture(0)
while(1):
    ret, sframe = cap.read()
    frame = cv2.resize(sframe, (28, 28), cv2.INTER_LINEAR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dat = frame.reshape(28, 28, 1) / 255.0
    res = np.argmax(model.predict(np.array([dat,])))
    print(res)
    cv2.imshow("capture", sframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

'''
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

for i in range(0, 100):
    res = model.predict(x_train[i:i+1])
    #print(np.array([x_train[i:i+1][0],]).shape)
    print(np.argmax(res[0]), y_train[i])
'''