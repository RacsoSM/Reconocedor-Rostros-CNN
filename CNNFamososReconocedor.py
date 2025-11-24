import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import pickle
import os

# --- CONFIGURACIÓN DEL PROYECTO ---
model_path = "modelo_facial_resnet_turbo.h5" 
label_path = "etiquetas.pickle"
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
UMBRAL_CONFIANZA = 85.0 # Si la certeza es menor a esto, es Desconocido

# 1. Cargar modelo y etiquetas
print("[INFO] Cargando modelo entrenado...")
try:
    model = load_model(model_path)
    with open(label_path, "rb") as f:
        labels_dict = pickle.load(f)
        classes = {v: k for k, v in labels_dict.items()}
    print(f"[INFO] Sistema listo. Clases cargadas: {len(classes)}")
except Exception as e:
    print(f"❌ ERROR CRÍTICO: {e}")
    exit()

# 2. Cargar Detector OpenCV (Haar Cascade)
# Cumple con el requisito de "Detección con OpenCV"
face_cascade = cv2.CascadeClassifier(cascade_path)

# 3. Iniciar cámara
print("[INFO] Iniciando cámara...")
cap = cv2.VideoCapture(0) # <--- AJUSTA EL ID DE TU CÁMARA (0, 1, 2 o 3)

# Resolución HD (Mejora la detección de OpenCV)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

window_name = "Sistema de Reconocimiento Facial (OpenCV)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Preparar imagen para Haar Cascade (Escala de Grises)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- DETECCIÓN FACIAL AJUSTADA (EL TRUCO ANTI-FANTASMAS) ---
    # scaleFactor=1.1: Escala de pirámide (estándar)
    # minNeighbors=8: (Antes 5). Exige que 8 cuadros detecten la cara. Elimina lámparas.
    # minSize=(100, 100): Ignora cuadros muy pequeños que suelen ser errores.
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=8,  
        minSize=(100, 100) 
    )

    for (x, y, w, h) in faces:
        # --- PREPROCESAMIENTO ---
        face_roi = frame[y:y+h, x:x+w]
        
        try:
            # Conversión de Color: BGR (OpenCV) -> RGB (Red Neuronal)
            roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            roi_resized = cv2.resize(roi_rgb, (224, 224))
            roi_array = img_to_array(roi_resized)
            roi_array = preprocess_input(roi_array) # Escala de -1 a 1
            roi_array = np.expand_dims(roi_array, axis=0)

            # --- INFERENCIA ---
            preds = model.predict(roi_array, verbose=0)[0]
            idx = np.argmax(preds)
            confidence = preds[idx] * 100
            label = classes.get(idx, "Error")

            # --- LÓGICA DE SEMÁFORO (ROJO/VERDE) ---
            
            # 1. ¿Es la clase "Z_Desconocidos"?
            es_carpeta_desconocidos = "Z_Desconocidos" in label
            
            # 2. ¿La confianza es muy baja?
            es_baja_confianza = confidence < UMBRAL_CONFIANZA

            if es_carpeta_desconocidos or es_baja_confianza:
                # CASO ROJO: Es un desconocido o no estamos seguros
                texto_mostrar = "DESCONOCIDO"
                color_box = (0, 0, 255) # Rojo
                
                # (Opcional) Si quieres ver por qué falló en consola:
                # print(f"DEBUG: Etiqueta={label}, Confianza={confidence:.1f}%")
                
            else:
                # CASO VERDE: Es un compañero o tú con alta seguridad
                texto_mostrar = f"{label}: {confidence:.0f}%"
                color_box = (0, 255, 0) # Verde

            # --- DIBUJAR EN PANTALLA ---
            # Rectángulo
            cv2.rectangle(frame, (x, y), (x+w, y+h), color_box, 2)
            
            # Etiqueta con fondo para legibilidad
            cv2.rectangle(frame, (x, y - 35), (x + w, y), color_box, -1)
            cv2.putText(frame, texto_mostrar, (x + 5, y - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        except Exception as e:
            pass # Si hay error en el recorte, ignoramos este frame

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()