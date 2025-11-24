import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
# CAMBIO: Agregamos BatchNormalization y ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import os

# --- CONFIGURACIÓN ---
dataset_path = r"C:\Users\OSCAR\Desktop\Tec\7mo\IA\CNN\Dataset\FamososMini"
# CAMBIÉ EL NOMBRE PARA QUE SEPAS QUE ES LA VERSIÓN POTENTE
model_save_path = "modelo_facial_resnet.h5" 
label_save_path = "etiquetas.pickle"

# Aumentamos un poco las épocas porque usamos Learning Rate dinámico
EPOCHS_FASE_1 = 25 
EPOCHS_FASE_2 = 20 
BS = 32
INIT_LR = 1e-4

# 1. Preparar Data Augmentation
# Reduje un poco la rotación (15) porque las caras suelen estar derechas
aug = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
    preprocessing_function=preprocess_input 
)

print("[INFO] Cargando imágenes...")
train_generator = aug.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=BS,
    class_mode="categorical",
    shuffle=True
)

class_names = train_generator.class_indices
print(f"[INFO] Clases encontradas: {class_names}")
with open(label_save_path, "wb") as f:
    pickle.dump(class_names, f)

# 2. Construir la Red Neuronal (ResNet50V2)
print("[INFO] Descargando ResNet50V2...")
baseModel = ResNet50V2(weights="imagenet", include_top=False,
                       input_tensor=Input(shape=(224, 224, 3)))

# FASE 1: Congelar base
baseModel.trainable = False

headModel = baseModel.output
headModel = GlobalAveragePooling2D()(headModel)

# --- MEJORA 1: ARQUITECTURA MÁS ROBUSTA ---
headModel = BatchNormalization()(headModel) # Estabiliza el aprendizaje
headModel = Dense(512, activation="relu")(headModel) # Más neuronas (256 -> 512)
headModel = Dropout(0.4)(headModel) # Dropout un poco más bajo para aprender más
headModel = Dense(len(class_names), activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# --- MEJORA 2: CALLBACKS INTELIGENTES ---
# Si el loss no baja en 3 vueltas, reduce la velocidad a la mitad (factor=0.5)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# 3. Compilar y Entrenar FASE 1
print("\n--- FASE 1: Calentamiento (Head Training) ---")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

model.fit(
    train_generator,
    epochs=EPOCHS_FASE_1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# --- FASE 2: DEEP FINE TUNING ---
print("\n--- FASE 2: Deep Fine Tuning (AQUÍ SUBE EL ACCURACY) ---")

baseModel.trainable = True

# --- MEJORA 3: DESCONGELAR MÁS CAPAS ---
# Antes era 150. Ahora 100. Dejamos que aprenda casi la mitad de la red.
fine_tune_at = 170
for layer in baseModel.layers[:fine_tune_at]:
    layer.trainable = False

# Learning rate bajo para no romper lo aprendido
opt_fine = Adam(learning_rate=1e-5) 
model.compile(loss="categorical_crossentropy", optimizer=opt_fine, metrics=["accuracy"])

model.fit(
    train_generator,
    epochs=EPOCHS_FASE_2,
    callbacks=[early_stop, reduce_lr], # Usamos el reductor de velocidad aquí también
    verbose=1
)

# 5. Guardar el modelo
print(f"[INFO] Guardando modelo PRO en {model_save_path}...")
model.save(model_save_path)
print("¡Entrenamiento PRO completado!")