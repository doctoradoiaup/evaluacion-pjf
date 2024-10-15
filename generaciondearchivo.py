# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:57:43 2024

@author: jperezr
"""

from sklearn.linear_model import LinearRegression
import pickle

# Suponiendo que tienes datos de entrenamiento
X_train = ...  # tus características
y_train = ...  # tus etiquetas

model = LinearRegression()
model.fit(X_train, y_train)

# Guardar el modelo
with open('modelo_calificacion.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)