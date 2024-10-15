# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:17:42 2024

@author: jperezr
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:57:43 2024

@author: jperezr
"""

from sklearn.linear_model import LinearRegression
import pickle
import numpy as np

# Supongamos que tienes un conjunto de datos de ejemplo
# Aquí se generan datos aleatorios para la demostración.
# Reemplaza esto con tus datos reales.

# Datos de ejemplo: 10 muestras y 2 características
np.random.seed(0)  # Para reproducibilidad
X_train = np.random.rand(10, 2)  # 10 muestras, 2 características
y_train = np.random.rand(10)  # 10 etiquetas (calificaciones)

# Entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Guardar el modelo
with open('modelo_calificacion.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Modelo guardado como 'modelo_calificacion.pkl'")