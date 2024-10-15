# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:59:06 2024

@author: jperezr
"""

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("Evaluación de Candidatos a Jueces y Magistrados")
st.header("Sistema de Calificación de Ensayos y Cartas de Referencia - Nueva Reforma al Poder Judicial de la Federación con Calificación Académica de Licenciatura")


# Función para obtener la calificación del modelo de PLN
def get_score_from_nlp_model(text):
    with open('modelo_calificacion.pkl', 'rb') as model_file:
        vectorizer, model = pickle.load(model_file)  # Extraer la tupla en dos variables
    
    # Transformar el texto usando el vectorizador
    text_array = vectorizer.transform([text])  # Transformar el texto a la forma que espera el modelo
    score = model.predict(text_array)  # Ahora model es el clasificador que puedes usar
    return score[0]  # Devuelve la calificación predicha

# Cargar ensayos y cartas de referencia
uploaded_files_essay = st.file_uploader("Sube los 10 archivos de ensayo", type=["txt"], accept_multiple_files=True)
uploaded_files_reference = st.file_uploader("Sube los 10 archivos de referencia", type=["txt"], accept_multiple_files=True)
uploaded_file_csv = st.file_uploader("Sube el archivo CSV", type=["csv"])

# Verifica que se han subido los archivos
if uploaded_files_essay and uploaded_files_reference and uploaded_file_csv:
    candidates = []

    # Leer CSV y extraer promedios académicos
    df_academics = pd.read_csv(uploaded_file_csv, encoding='ISO-8859-1')  # Cambia a 'latin1' si es necesario
    df_academics.columns = df_academics.columns.str.strip()  # Quitar espacios en los nombres de columnas
    df_academics = df_academics.iloc[:, [0, 2]]  # Tomar la primera y la tercera columna (nombre y promedio académico)
    df_academics.columns = ['Nombre', 'Calificación Licenciatura']  # Renombrar las columnas

    # Depurar nombres para asegurarnos de que no haya espacios adicionales
    df_academics['Nombre'] = df_academics['Nombre'].str.strip()

    # Calificaciones
    essay_scores = []
    reference_scores = []
    
    for essay_file, reference_file in zip(uploaded_files_essay, uploaded_files_reference):
        # Extraer el nombre del candidato eliminando el prefijo correspondiente
        candidate_name_essay = essay_file.name.replace('ensayo-', '').replace('.txt', '').strip()  # Extraer nombre del archivo de ensayo
        candidate_name_reference = reference_file.name.replace('referencia-', '').replace('.txt', '').strip()  # Extraer nombre del archivo de referencia
        
        # Asegurarse de que ambos nombres coincidan
        if candidate_name_essay != candidate_name_reference:
            st.error(f"Error: El nombre del archivo de ensayo '{candidate_name_essay}' no coincide con el nombre del archivo de referencia '{candidate_name_reference}'")
            continue
        
        candidates.append(candidate_name_essay)

        # Leer contenido de los archivos
        essay_content = essay_file.read().decode("utf-8")
        reference_content = reference_file.read().decode("utf-8")

        # Obtener calificaciones
        essay_score = get_score_from_nlp_model(essay_content)
        reference_score = get_score_from_nlp_model(reference_content)

        essay_scores.append(essay_score)
        reference_scores.append(reference_score)

    # Crear DataFrame con las calificaciones
    df_scores = pd.DataFrame({
        'Nombre': candidates,
        'Calificación Ensayo': essay_scores,
        'Calificación Carta de Referencia': reference_scores,
    })

    # Agregar calificación de licenciatura
    df_scores = df_scores.merge(df_academics, on='Nombre', how='left')

    # Mostrar nombres de candidatos y calificaciones para depuración
    st.write("Candidatos y calificaciones del CSV:")
    st.write(df_academics)

    st.write("Candidatos y calificaciones calculadas:")
    st.write(df_scores)

    # Calificación total
    df_scores['Calificación Total'] = df_scores[['Calificación Ensayo', 'Calificación Carta de Referencia', 'Calificación Licenciatura']].mean(axis=1)

    # Mostrar DataFrame de calificaciones
    st.subheader("Calificación Total de Candidatos")
    st.dataframe(df_scores)

    # Gráficos de calificaciones
    st.subheader("Gráficos de Calificaciones")
    
    # Gráfico de calificación de ensayo
    plt.figure(figsize=(10, 5))
    plt.bar(df_scores['Nombre'], df_scores['Calificación Ensayo'], color='blue')
    plt.title('Calificación de Ensayo')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Gráfico de calificación de carta de referencia
    plt.figure(figsize=(10, 5))
    plt.bar(df_scores['Nombre'], df_scores['Calificación Carta de Referencia'], color='green')
    plt.title('Calificación de Carta de Referencia')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Gráfico de calificación total
    plt.figure(figsize=(10, 5))
    plt.bar(df_scores['Nombre'], df_scores['Calificación Total'], color='orange')
    plt.title('Calificación Total')
    plt.xticks(rotation=45)
    st.pyplot(plt)