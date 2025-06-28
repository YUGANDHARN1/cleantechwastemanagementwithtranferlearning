"""
VGG16 Transfer Learning Model for Waste Classification
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

class WasteClassificationModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_names = ['Recyclable', 'Organic', 'Hazardous', 'Non-Recyclable']
        self.model = None
        self.model_path = 'vgg16.h5'
        
    def create_model(self):
        """Create VGG16 transfer learning model"""
        # Load pre-trained VGG16 model
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Add custom classification layers
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', name='dense_1')(x)
        x = Dropout(0.5, name='dropout_1')(x)
        x = Dense(256, activation='relu', name='dense_2')(x)
        x = Dropout(0.3, name='dropout_2')(x)
        outputs = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        self.model = Model(inputs, outputs, name='waste_classification_vgg16')
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def load_model(self):
        """Load existing model or create new one"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Model loaded from {self.model_path}")
            else:
                print("Model file not found. Creating new model...")
                self.create_model()
                print("New model created successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model as fallback...")
            self.create_model()
    
    def save_model(self, path=None):
        """Save the model"""
        if path is None:
            path = self.model_path
        
        if self.model is not None:
            self.model.save(path)
            print(f"Model saved to {path}")
        else:
            print("No model to save")
    
    def predict(self, image_array):
        """Make prediction on preprocessed image"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Ensure image has batch dimension
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image_array, verbose=0)
        
        return predictions[0]  # Return probabilities for single image
    
    def get_top_predictions(self, predictions, top_k=4):
        """Get top k predictions with class names"""
        # Get indices sorted by probability (descending)
        top_indices = np.argsort(predictions)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'category': self.class_names[idx],
                'confidence': float(predictions[idx]) * 100,
                'probability': float(predictions[idx])
            })
        
        return results
    
    def fine_tune(self, trainable_layers=10):
        """Fine-tune the model by unfreezing some layers"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Get the base VGG16 model
        base_model = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.applications.VGG16):
                base_model = layer
                break
        
        if base_model is None:
            print("VGG16 base model not found for fine-tuning")
            return
        
        # Unfreeze the top layers
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) - trainable_layers
        
        # Freeze all layers before fine_tune_at
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model prepared for fine-tuning. {trainable_layers} layers unfrozen.")
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is not None:
            return self.model.summary()
        else:
            return "Model not loaded"
    
    def preprocess_image(self, image):
        """Preprocess image for VGG16 model"""
        # Resize to model input size
        image = image.resize((self.input_shape[0], self.input_shape[1]))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array / 255.0
        
        return img_array
    
    def create_data_generators(self, train_dir, validation_dir, batch_size=32):
        """Create data generators for training"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.class_names
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical',
            classes=self.class_names
        )
        
        return train_generator, validation_generator
    
    def train(self, train_generator, validation_generator, epochs=10):
        """Train the model"""
        if self.model is None:
            self.create_model()
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                self.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

# Example usage and testing
if __name__ == "__main__":
    # Create model instance
    model = WasteClassificationModel()
    
    # Load or create model
    model.load_model()
    
    # Print model summary
    if model.model:
        print("Model Summary:")
        model.get_model_summary()
        
        print(f"\nModel input shape: {model.model.input_shape}")
        print(f"Model output shape: {model.model.output_shape}")
        print(f"Total parameters: {model.model.count_params():,}")
        
        # Test with dummy data
        dummy_input = np.random.random((1, 224, 224, 3))
        predictions = model.predict(dummy_input)
        top_predictions = model.get_top_predictions(predictions)
        
        print("\nTest prediction results:")
        for pred in top_predictions:
            print(f"- {pred['category']}: {pred['confidence']:.2f}%")
    else:
        print("Failed to create/load model")