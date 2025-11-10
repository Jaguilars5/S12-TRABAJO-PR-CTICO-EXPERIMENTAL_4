import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

print("Ejecutando Red Neuronal Simple (Backpropagation XOR)...")

# 1. Datos de entrenamiento (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 2. Crear red neuronal
model = Sequential(
    [
        Dense(4, activation="relu", input_shape=(2,)),  # capa oculta
        Dense(1, activation="sigmoid"),  # capa de salida
    ]
)

# 3. Compilar
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 4. Entrenar
model.fit(X, y, epochs=500, verbose=0)
print("Entrenamiento completado.")

# 5. Evaluar predicciones
print("\nPredicciones XOR:")
print("Entradas:\n", X)
print("\nPredicciones (0 = ~0, 1 = ~1):\n", model.predict(X).round())
