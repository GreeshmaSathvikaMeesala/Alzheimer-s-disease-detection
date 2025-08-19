import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetV2B0, ResNet50, DenseNet121
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import albumentations as A
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow_addons as tfa

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 0.001

class AdvancedPreprocessor:
    def __init__(self):
        self.transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.3),
        ])

    def preprocess_image(self, image):
        # Apply advanced augmentations
        augmented = self.transform(image=image)
        image = augmented['image']
        
        # Normalize
        image = preprocess_input(image)
        return image

class EnhancedModel:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.initialize_models()

    def initialize_models(self):
        # EfficientNetV2B0
        base_model1 = EfficientNetV2B0(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        model1 = self.create_model_architecture(base_model1)
        self.models['efficientnet'] = model1
        self.weights['efficientnet'] = 0.4

        # ResNet50
        base_model2 = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        model2 = self.create_model_architecture(base_model2)
        self.models['resnet'] = model2
        self.weights['resnet'] = 0.3

        # DenseNet121
        base_model3 = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        model3 = self.create_model_architecture(base_model3)
        self.models['densenet'] = model3
        self.weights['densenet'] = 0.3

    def create_model_architecture(self, base_model):
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def compile_models(self):
        for name, model in self.models.items():
            model.compile(
                optimizer=Adam(learning_rate=LEARNING_RATE),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )

    def train(self, train_generator, val_generator):
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=PATIENCE,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                f'best_model_{name}.h5',
                monitor='val_accuracy',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            )
        ]

        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=EPOCHS,
                callbacks=callbacks
            )

    def predict(self, image):
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(image)
            predictions.append(pred * self.weights[name])
        return np.sum(predictions, axis=0)
def load_and_preprocess_data(data_dir):
    print("Loading and preprocessing data...")
    preprocessor = AdvancedPreprocessor()
    # Create data generators with advanced augmentation
    datagen = ImageDataGenerator(
        preprocessing_function=preprocessor.preprocess_image,
        validation_split=0.2
    )
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    # Load validation data
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    return train_generator, val_generator
def evaluate_model(model, val_generator):
    print("Evaluating model...")
    
    # Get predictions from ensemble
    y_pred = model.predict(val_generator)
    y_true = val_generator.classes
    
    # Calculate metrics
    accuracy = np.mean((y_pred > 0.5) == y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred > 0.5).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    roc_auc = roc_auc_score(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred > 0.5))

def grad_cam(model, img_path, layer_name='top_conv'):
    # Load and preprocess image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    
    # Get predictions from ensemble
    preds = model.predict(img)
    
    # Generate Grad-CAM for each model in ensemble
    heatmaps = []
    for name, submodel in model.models.items():
        last_conv_layer = submodel.get_layer(layer_name)
        grad_model = Model(
            [submodel.inputs],
            [last_conv_layer.output, submodel.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            loss = predictions[:, 0]
        
        grads = tape.gradient(loss, conv_outputs)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        heatmap = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        
        # Resize heatmap
        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmaps.append(heatmap * model.weights[name])
    
    # Combine heatmaps
    final_heatmap = np.sum(heatmaps, axis=0)
    final_heatmap = np.uint8(255 * final_heatmap)
    final_heatmap = cv2.applyColorMap(final_heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    superimposed_img = final_heatmap * 0.4 + img[0]
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img

def main():
    # Specify your data directory
    data_dir = "path/to/your/data"
    
    # Load and preprocess data
    train_generator, val_generator = load_and_preprocess_data(data_dir)
    
    # Create and train model
    model = EnhancedModel()
    model.compile_models()
    model.train(train_generator, val_generator)
    
    # Evaluate model
    evaluate_model(model, val_generator)
    
    # Example Grad-CAM visualization
    sample_img_path = "path/to/sample/image.jpg"
    grad_cam_img = grad_cam(model, sample_img_path)
    
    # Save Grad-CAM visualization
    cv2.imwrite("grad_cam_visualization.jpg", grad_cam_img)

if __name__ == "__main__":
    main() 