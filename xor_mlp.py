import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Dataset de entrenamiento para puerta XOR
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
y_train = np.array([[0], [1], [1], [0]], dtype="float32")

# Modelo MLP
model = keras.Sequential()
model.add(layers.Dense(2, input_dim=2, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

# configuración del modelo
model.compile(
    optimizer=keras.optimizers.Adam(0.1),
    loss="mean_squared_error",
    metrics=["accuracy"],
)  # Asumo que querías agregar 'accuracy' aquí

# Entrenamiento
print("Entrenando modelo MLP-XOR...")
fit_history = model.fit(x_train, y_train, epochs=50, batch_size=4, verbose=0)
print("Entrenamiento completado.")

# model.summary()
loss_curve = fit_history.history["loss"]
# accuracy_curve = fit_history.history['accuracy']

# plt.plot(accuracy_curve, label = 'precisión')
plt.plot(loss_curve, label="Pérdida")
plt.legend(loc="lower left")
plt.legend()
plt.title("Resultado del Entrenamiento (MLP-XOR)")
plt.show()

# Recuperamos bias and weights de la capa oculta
weights_HL, biases_HL = model.layers[0].get_weights()
# Recuperamos bias and weights de la capa de salida
weights_OL, biases_OL = model.layers[1].get_weights()

print("\nPesos Capa Oculta:\n", weights_HL)
print("\nBias Capa Oculta:\n", biases_HL)
print("\nPesos Capa Salida:\n", weights_OL)
print("\nBias Capa Salida:\n", biases_OL)

prediccion = model.predict(x_train)
print("\nEntradas:\n", x_train)
print("\nSalidas Reales:\n", y_train)
print("\nPredicciones:\n", prediccion.round())
