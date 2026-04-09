import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import mixed_precision

# ==========================
# CONFIG
# ==========================
DATASET_PATH = "dataset/train/tomato_only"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 12











0











SEED = 42

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

print("✅ Dataset exists:", os.path.exists(DATASET_PATH))

# ==========================
# GPU CHECK
# ==========================
gpus = tf.config.list_physical_devices("GPU")
print("GPUs Found:", gpus)

if gpus:
    print("✅ Training will use GPU")
else:
    print("⚠️ No GPU detected, training will run on CPU (slow)")

# ==========================
# MIXED PRECISION (FAST TRAINING)
# ==========================
mixed_precision.set_global_policy("mixed_float16")
print("✅ Mixed precision enabled:", mixed_precision.global_policy())

# ==========================
# LOAD DATASET (FAST tf.data)
# ==========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print("\n✅ Classes Detected:")
for i, c in enumerate(class_names):
    print(i, "->", c)

print("\n✅ Total Classes:", NUM_CLASSES)

# ==========================
# PERFORMANCE OPTIMIZATION
# ==========================
AUTOTUNE = tf.data.AUTOTUNE

# Avoid heavy cache RAM issues (Windows)
train_ds = train_ds.shuffle(300).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==========================
# DATA AUGMENTATION (STRONG FOR REAL IMAGES)
# ==========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])

# ==========================
# MODEL (EfficientNetV2 Modern Backbone)
# ==========================
base_model = tf.keras.applications.EfficientNetV2B0(
    include_top=False,
    input_shape=(224, 224, 3),
    weights="imagenet"
)

base_model.trainable = False  # fast MVP training

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet_v2.preprocess_input(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

# IMPORTANT: dtype float32 output because mixed precision
outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

model = tf.keras.Model(inputs, outputs)

# ==========================
# COMPILE (Modern Optimizer)
# ==========================
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n==================== MODEL SUMMARY ====================")
model.summary()

# ==========================
# CALLBACKS
# ==========================
callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "tomato_best.keras"),
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
]

# ==========================
# TRAIN (STAGE 1: HEAD TRAINING)
# ==========================
print("\n🚀 Training Stage 1: Feature Extractor (Frozen Backbone)")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================
# FINE-TUNING (STAGE 2)
# ==========================
print("\n🔥 Fine-tuning Stage 2: Unfreezing last 30 layers")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks
)

# ==========================
# SAVE FINAL MODEL
# ==========================
model.save(os.path.join(MODEL_DIR, "tomato_model_final.keras"))
print("\n✅ Tomato MVP model training completed and saved!")

# ==========================
# EXPORT TFLITE (MOBILE READY)
# ==========================
print("\n📦 Exporting TFLite model...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(os.path.join(MODEL_DIR, "tomato_model.tflite"), "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved: model/tomato_model.tflite")

# ==========================
# SAVE CLASS NAMES
# ==========================
with open(os.path.join(MODEL_DIR, "tomato_classes.txt"), "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("✅ Class names saved: model/tomato_classes.txt")