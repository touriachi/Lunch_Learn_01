
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from utilities import *



X_train, y_train, X_test, y_test = load_data()
X_train = X_train / 255
X_test = X_test / 255

plt.figure(figsize=(16, 8))
for i in range(1, 10):
    plt.subplot(4, 5, i)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.tight_layout()
plt.show()


X_train = X_train / 255
X_test = X_test / 255


print('trainset:', X_train) # 60,000 images
print('testset:', X_test) # 10,000 images

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(64,64)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])



# Compilation du modele
model.compile(optimizer='adam',
              loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)


# Evaluation du modele
test_loss, test_acc = model.evaluate(X_test,  y_test)
print('Test accuracy:', test_acc)

prediction_model = keras.Sequential([model, keras.layers.Softmax()])
predict_proba = prediction_model.predict(X_test)
predictions = np.argmax(predict_proba, axis=1)

print(predictions[:10])
print(y_test[:10])
