# ==================================================================================
# FACIAL LANDMARK DETECTION PROJECT
# Using Kaggle's Facial Keypoints Detection Dataset
# 
# Dataset: https://www.kaggle.com/c/facial-keypoints-detection
# This project detects 15 facial keypoints (30 coordinates) using CNN
# ==================================================================================

# Import all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, Flatten, Conv2D, MaxPool2D, 
                                   BatchNormalization, Input, Add, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import warnings
from PIL import Image
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("=" * 80)
print("FACIAL LANDMARK DETECTION PROJECT")
print("=" * 80)

# ==================================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# ==================================================================================

def load_kaggle_data():
    
    print("Loading Kaggle Facial Keypoints Detection dataset...")
    
    try:
        # Load training data
        print("Looking for 'training.csv' in:", os.getcwd())
        train_df = pd.read_csv('training.csv')
        train_df = train_df.sample(n=1000, random_state=42)  # using 1000 samples

        print(f"Training data loaded successfully: {train_df.shape}")
        
        # Load test data (optional)
        try:
            test_df = pd.read_csv('test.csv')
            print(f"Test data loaded successfully: {test_df.shape}")
        except:
            print("Test data not found, will create validation split from training data")
            test_df = None
            
        return train_df, test_df
        
    except FileNotFoundError:
        print("Dataset files not found!")
        print("Please download the dataset from: https://www.kaggle.com/c/facial-keypoints-detection")
        print("Creating sample data for demonstration...")
        return create_sample_data()

def create_sample_data():
    
    print("Creating sample data for demonstration...")
    
    # Create sample facial keypoint data
    n_samples = 2000
    
    # 15 facial keypoints = 30 coordinates (x,y pairs)
    keypoint_columns = [
        'left_eye_center_x', 'left_eye_center_y',
        'right_eye_center_x', 'right_eye_center_y',
        'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
        'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
        'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
        'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
        'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
        'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
        'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
        'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
        'nose_tip_x', 'nose_tip_y',
        'mouth_left_corner_x', 'mouth_left_corner_y',
        'mouth_right_corner_x', 'mouth_right_corner_y',
        'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
        'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y'
    ]
    
    # Generate realistic facial keypoint coordinates
    sample_data = {}
    
    # Eyes (around position 30-65, 35-50)
    for eye_point in ['left_eye_center', 'right_eye_center', 'left_eye_inner_corner', 
                      'left_eye_outer_corner', 'right_eye_inner_corner', 'right_eye_outer_corner']:
        sample_data[f'{eye_point}_x'] = np.random.normal(48, 15, n_samples)
        sample_data[f'{eye_point}_y'] = np.random.normal(42, 8, n_samples)
    
    # Eyebrows (slightly above eyes)
    for brow_point in ['left_eyebrow_inner_end', 'left_eyebrow_outer_end',
                       'right_eyebrow_inner_end', 'right_eyebrow_outer_end']:
        sample_data[f'{brow_point}_x'] = np.random.normal(48, 15, n_samples)
        sample_data[f'{brow_point}_y'] = np.random.normal(35, 8, n_samples)
    
    # Nose (center of face)
    sample_data['nose_tip_x'] = np.random.normal(48, 5, n_samples)
    sample_data['nose_tip_y'] = np.random.normal(58, 8, n_samples)
    
    # Mouth (lower part of face)
    for mouth_point in ['mouth_left_corner', 'mouth_right_corner', 
                        'mouth_center_top_lip', 'mouth_center_bottom_lip']:
        sample_data[f'{mouth_point}_x'] = np.random.normal(48, 10, n_samples)
        sample_data[f'{mouth_point}_y'] = np.random.normal(75, 8, n_samples)
    
    # Clip coordinates to valid range [0, 96]
    for key in sample_data:
        sample_data[key] = np.clip(sample_data[key], 0, 96)
    
    # Add some missing values (like real dataset)
    for key in sample_data:
        mask = np.random.random(n_samples) < 0.1  # 10% missing values
        sample_data[key][mask] = np.nan
    
    # Generate sample images (96x96 grayscale)
    images = []
    for i in range(n_samples):
        # Create a simple face-like image
        img = np.random.randint(0, 256, (96, 96), dtype=np.uint8)
        # Add some structure to make it look more face-like
        img[20:80, 20:80] = np.random.randint(100, 200, (60, 60))  # Face region
        img[35:50, 35:60] = np.random.randint(50, 150, (15, 25))   # Eye region
        img[60:70, 40:56] = np.random.randint(80, 180, (10, 16))   # Mouth region
        images.append(' '.join(map(str, img.flatten())))
    
    sample_data['Image'] = images
    
    train_df = pd.DataFrame(sample_data)
    
    print(f"Sample training data created: {train_df.shape}")
    return train_df, None

def preprocess_data(train_df, test_df=None):
    """
    Preprocess the facial keypoints data
    """
    print("Preprocessing data...")
    
    # Get keypoint column names (all except 'Image')
    keypoint_cols = [col for col in train_df.columns if col != 'Image']
    
    # Display dataset info
    print(f"Dataset shape: {train_df.shape}")
    print(f"Number of keypoint features: {len(keypoint_cols)}")
    print(f"Number of images: {len(train_df)}")
    
    # Check for missing values
    print("\nMissing values per feature:")
    missing_counts = train_df[keypoint_cols].isnull().sum()
    print(missing_counts[missing_counts > 0])
    
    # Handle missing values - fill with mean
    print("Filling missing values with mean...")
    train_df[keypoint_cols] = train_df[keypoint_cols].fillna(train_df[keypoint_cols].mean())
    
    # Convert image strings to numpy arrays
    print("Converting image data...")
    def string_to_image(img_string):
        img = np.array(img_string.split(), dtype=np.float32)
        return img.reshape(96, 96, 1)
    
    # Process training images
    train_images = np.array([string_to_image(img) for img in train_df['Image']])
    train_images = train_images / 255.0  # Normalize to [0, 1]
    
    # Extract keypoints and normalize to [0, 1]
    train_keypoints = train_df[keypoint_cols].values.astype(np.float32)
    train_keypoints = train_keypoints / 96.0  # Normalize to [0, 1]
    
    print(f"Training images shape: {train_images.shape}")
    print(f"Training keypoints shape: {train_keypoints.shape}")
    
    # Process test data if available
    test_images = None
    if test_df is not None:
        test_images = np.array([string_to_image(img) for img in test_df['Image']])
        test_images = test_images / 255.0
        print(f"Test images shape: {test_images.shape}")
    
    return train_images, train_keypoints, test_images

# ==================================================================================
# STEP 2: DATA VISUALIZATION
# ==================================================================================

def plot_sample_images(images, keypoints, num_samples=6):
    
    print("Plotting sample images with keypoints...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(num_samples):
        # Display image
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        
        # Plot keypoints
        kp = keypoints[i] * 96  # Denormalize keypoints
        
        # Plot each keypoint as a red dot
        for j in range(0, len(kp), 2):
            if j+1 < len(kp) and not np.isnan(kp[j]) and not np.isnan(kp[j+1]):
                axes[i].plot(kp[j], kp[j+1], 'ro', markersize=4)
        
        axes[i].set_title(f'Sample Image {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_keypoint_distribution(keypoints):
    """
    Plot distribution of keypoint coordinates
    """
    print("Plotting keypoint distribution...")
    
    # Denormalize for visualization
    kp_denorm = keypoints * 96
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot x-coordinates
    ax1.hist(kp_denorm[:, ::2].flatten(), bins=30, alpha=0.7, color='blue')
    ax1.set_title('X-coordinates Distribution')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Frequency')
    
    # Plot y-coordinates
    ax2.hist(kp_denorm[:, 1::2].flatten(), bins=30, alpha=0.7, color='red')
    ax2.set_title('Y-coordinates Distribution')
    ax2.set_xlabel('Y coordinate')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# ==================================================================================
# STEP 3: MODEL ARCHITECTURE
# ==================================================================================

def create_basic_cnn_model(input_shape, num_keypoints):
    """
    Create a basic CNN model for facial keypoint detection
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPool2D(2, 2),
        Dropout(0.2),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(2, 2),
        Dropout(0.2),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(2, 2),
        Dropout(0.3),
        
        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(2, 2),
        Dropout(0.3),
        
        # Fifth Convolutional Block
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(2, 2),
        Dropout(0.4),
        
        # Fully Connected Layers
        Flatten(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output layer (linear activation for regression)
        Dense(num_keypoints, activation='linear')
    ])
    
    return model

def create_advanced_cnn_model(input_shape, num_keypoints):
    """
    Create an advanced CNN model with residual connections
    """
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv2D(32, (7, 7), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPool2D(2, 2)(x)
    
    # Residual Block 1
    shortcut = x
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Match dimensions for residual connection
    shortcut = Conv2D(64, (1, 1), padding='same')(shortcut)
    x = Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    x = MaxPool2D(2, 2)(x)
    x = Dropout(0.25)(x)
    
    # Residual Block 2
    shortcut = x
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    shortcut = Conv2D(128, (1, 1), padding='same')(shortcut)
    x = Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    x = MaxPool2D(2, 2)(x)
    x = Dropout(0.25)(x)
    
    # Residual Block 3
    shortcut = x
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    shortcut = Conv2D(256, (1, 1), padding='same')(shortcut)
    x = Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    x = MaxPool2D(2, 2)(x)
    x = Dropout(0.3)(x)
    
    # Global Average Pooling instead of Flatten
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    outputs = Dense(num_keypoints, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# ==================================================================================
# STEP 4: TRAINING UTILITIES
# ==================================================================================

def create_data_augmentation():
    """
    Create data augmentation generator
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # Don't flip faces horizontally
        fill_mode='nearest'
    )
    return datagen

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Train the facial keypoint detection model
    """
    print("Compiling model...")
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Print model summary
    print(model.summary())
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=8,
        min_lr=1e-7,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_facial_keypoints_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    callbacks = [early_stopping, reduce_lr, model_checkpoint]
    
    print("Starting training...")
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """
    Plot training and validation metrics
    """
    print("Plotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation loss
    ax1.plot(history.history['loss'], label='Training Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training & validation MAE
    ax2.plot(history.history['mae'], label='Training MAE', color='blue')
    ax2.plot(history.history['val_mae'], label='Validation MAE', color='red')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# ==================================================================================
# STEP 5: MODEL EVALUATION
# ==================================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model
    """
    print("Evaluating model on test data...")
    
    # Make predictions
    predictions = model.predict(X_test, verbose=1)
    
    # Calculate metrics
    mse = np.mean((predictions - y_test) ** 2)
    mae = np.mean(np.abs(predictions - y_test))
    rmse = np.sqrt(mse)
    
    print(f"\nTest Results:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE in pixels (denormalized): {mae * 96:.2f}")
    
    return predictions

def visualize_predictions(images, true_keypoints, predicted_keypoints, num_samples=8):
    """
    Visualize model predictions vs ground truth
    """
    print("Visualizing predictions...")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for i in range(num_samples):
        row = i // 4
        col = i % 4
        
        # Display image
        axes[row, col].imshow(images[i].squeeze(), cmap='gray')
        
        # Plot ground truth keypoints (green)
        true_kp = true_keypoints[i] * 96  # Denormalize
        for j in range(0, len(true_kp), 2):
            if j+1 < len(true_kp):
                axes[row, col].plot(true_kp[j], true_kp[j+1], 'go', markersize=4, label='Ground Truth' if j == 0 else "")
        
        # Plot predicted keypoints (red)
        pred_kp = predicted_keypoints[i] * 96  # Denormalize
        for j in range(0, len(pred_kp), 2):
            if j+1 < len(pred_kp):
                axes[row, col].plot(pred_kp[j], pred_kp[j+1], 'rx', markersize=4, label='Prediction' if j == 0 else "")
        
        axes[row, col].set_title(f'Sample {i+1}')
        axes[row, col].axis('off')
        if i == 0:
            axes[row, col].legend()
    
    plt.tight_layout()
    plt.show()

def calculate_keypoint_accuracy(true_keypoints, predicted_keypoints, threshold=5):
    """
    Calculate accuracy based on pixel distance threshold
    """
    # Denormalize keypoints
    true_kp = true_keypoints * 96
    pred_kp = predicted_keypoints * 96
    
    # Calculate distances
    distances = np.sqrt((true_kp[:, ::2] - pred_kp[:, ::2])**2 + 
                       (true_kp[:, 1::2] - pred_kp[:, 1::2])**2)
    
    # Calculate accuracy (percentage of keypoints within threshold)
    accuracy = np.mean(distances < threshold) * 100
    
    print(f"Keypoint Accuracy (within {threshold} pixels): {accuracy:.2f}%")
    print(f"Average distance error: {np.mean(distances):.2f} pixels")
    
    return accuracy

# ==================================================================================
# STEP 6: MAIN EXECUTION PIPELINE
# ==================================================================================

def main():
    """
    Main function to run the complete facial landmark detection pipeline
    """
    print("STARTING FACIAL LANDMARK DETECTION PROJECT")
    print("=" * 80)
    
    # Step 1: Load Data
    print("\n1. LOADING DATA")
    print("-" * 40)
    train_df, test_df = load_kaggle_data()
    
    # Step 2: Preprocess Data
    print("\n2. PREPROCESSING DATA")
    print("-" * 40)
    train_images, train_keypoints, test_images = preprocess_data(train_df, test_df)
    
    # Step 3: Visualize Data
    print("\n3. VISUALIZING DATA")
    print("-" * 40)
    plot_sample_images(train_images, train_keypoints)
    plot_keypoint_distribution(train_keypoints)
    
    # Step 4: Split Data
    print("\n4. SPLITTING DATA")
    print("-" * 40)
    X_train, X_val, y_train, y_val = train_test_split(
        train_images, train_keypoints, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Step 5: Create Model
    print("\n5. CREATING MODEL")
    print("-" * 40)
    input_shape = X_train.shape[1:]
    num_keypoints = y_train.shape[1]
    
    print(f"Input shape: {input_shape}")
    print(f"Number of keypoints: {num_keypoints}")
    
    # Choose model architecture
    print("Creating CNN model...")
    model = create_basic_cnn_model(input_shape, num_keypoints)
    
    # Alternative: Use advanced model
    # model = create_advanced_cnn_model(input_shape, num_keypoints)
    
    # Step 6: Train Model
    print("\n6. TRAINING MODEL")
    print("-" * 40)
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=80, batch_size=32)
    
    # Step 7: Plot Training History
    print("\n7. PLOTTING TRAINING RESULTS")
    print("-" * 40)
    plot_training_history(history)
    
    # Step 8: Evaluate Model
    print("\n8. EVALUATING MODEL")
    print("-" * 40)
    predictions = evaluate_model(model, X_val, y_val)
    
    # Step 9: Calculate Accuracy
    print("\n9. CALCULATING ACCURACY")
    print("-" * 40)
    calculate_keypoint_accuracy(y_val, predictions)
    
    # Step 10: Visualize Results
    print("\n10. VISUALIZING RESULTS")
    print("-" * 40)
    visualize_predictions(X_val, y_val, predictions)
    
    # Step 11: Save Model
    print("\n11. SAVING MODEL")
    print("-" * 40)
    model.save('final_facial_keypoints_model.h5')
    print("Model saved as 'final_facial_keypoints_model.h5'")
    
    # Step 12: Model Summary
    print("\n12. PROJECT SUMMARY")
    print("-" * 40)
    print(f"✓ Successfully trained facial landmark detection model")
    print(f"✓ Model architecture: CNN with {len(model.layers)} layers")
    print(f"✓ Training samples: {X_train.shape[0]}")
    print(f"✓ Validation samples: {X_val.shape[0]}")
    print(f"✓ Number of keypoints detected: {num_keypoints // 2}")
    print(f"✓ Final validation loss: {min(history.history['val_loss']):.6f}")
    print(f"✓ Final validation MAE: {min(history.history['val_mae']):.6f}")
    
    print("\n" + "=" * 80)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return model, history

# ==================================================================================
# ADDITIONAL UTILITIES
# ==================================================================================

def predict_single_image(model, image_path):
    """
    Predict keypoints for a single image
    """
    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (96, 96))
    img = img.reshape(1, 96, 96, 1) / 255.0
    
    # Predict keypoints
    prediction = model.predict(img)
    
    # Denormalize
    keypoints = prediction[0] * 96
    
    return keypoints

def create_submission_file(model, test_images, submission_path='submission.csv'):
    
    if test_images is not None:
        print("Creating submission file...")
        predictions = model.predict(test_images)
        
        
        submission_data = []
        for i, pred in enumerate(predictions):
            for j in range(0, len(pred), 2):
                submission_data.append({
                    'ImageId': i + 1,
                    'FeatureName': f'keypoint_{j//2}_x',
                    'Location': pred[j] * 96
                })
                submission_data.append({
                    'ImageId': i + 1,
                    'FeatureName': f'keypoint_{j//2}_y',
                    'Location': pred[j+1] * 96
                })
        
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission file saved as '{submission_path}'")


# ==================================================================================
# RUN MAIN FUNCTION
# ==================================================================================

if __name__ == "__main__":
    main()
