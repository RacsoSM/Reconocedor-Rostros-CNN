import cv2
import os
import time

# --- CONFIGURACIÓN ---
NOMBRE_CARPETA = r"C:\Users\saidr\OneDrive\Desktop\Said Rodriguez"
CANTIDAD_FOTOS = 150
ANCHO_FOTO = 640
ALTO_FOTO = 480
TIEMPO_ESPERA = 0.2

# 1. Crear carpeta
if not os.path.exists(NOMBRE_CARPETA):
    os.makedirs(NOMBRE_CARPETA)
    print(f"Carpeta creada: {NOMBRE_CARPETA}")

# 2. Iniciar cámara
cap = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
ultimo_tiempo = time.time()

print("INICIANDO CAPTURA LIMPIA (SIN CUADRO)...")
print(f"Se tomará una foto cada {TIEMPO_ESPERA} segundos.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.resize(frame, (ANCHO_FOTO, ALTO_FOTO))
    
    # --- CAMBIO CLAVE AQUÍ ---
    # Creamos una copia para dibujar en ella. 
    # 'frame' se queda limpia para guardar, 'frame_display' es para ver en pantalla.
    frame_display = frame.copy() 
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # Dibujar rectángulo verde EN LA COPIA (frame_display)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # --- LÓGICA DE CAPTURA ---
    if time.time() - ultimo_tiempo >= TIEMPO_ESPERA:
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            
            # Recortamos usando 'frame' (la ORIGINAL LIMPIA)
            rostro_recortado = frame[y:y+h, x:x+w]
            
            try:
                rostro_recortado = cv2.resize(rostro_recortado, (400, 400))
            except:
                continue
            
            count += 1
            file_name_path = f"{NOMBRE_CARPETA}/Usuario_{count}.jpg"
            cv2.imwrite(file_name_path, rostro_recortado)
            
            print(f"Foto HQ {count} guardada!")
            
            ultimo_tiempo = time.time()
            
            # Efecto flash (Visual en la copia)
            cv2.rectangle(frame_display, (0,0), (ANCHO_FOTO, ALTO_FOTO), (255,255,255), cv2.FILLED)
            cv2.imshow("Capturando Rostro", frame_display)
            cv2.waitKey(50) 

    # Poner texto en la COPIA
    cv2.putText(frame_display, f"Fotos HQ: {count}/{CANTIDAD_FOTOS}", (30, 50), 
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

    # Mostrar la COPIA
    cv2.imshow("Capturando Rostro", frame_display)

    if cv2.waitKey(1) == ord('q') or count >= CANTIDAD_FOTOS:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n¡Listo! {count} fotos de ALTA CALIDAD guardadas.")