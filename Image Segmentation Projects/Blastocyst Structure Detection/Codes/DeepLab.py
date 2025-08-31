# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 19:16:08 2024

@author: hlias
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses, layers, Model
from sklearn.model_selection import train_test_split

# Paths for images and masks
image_path = r'C:\Full\Path\To\Your\Images\Folder'
icm_mask_path = r'C:\Full\Path\To\Your\GT_ICM\Folder'
te_mask_path = r'C:\Full\Path\To\Your\GT_TE\Folder'
zp_mask_path = r'C:\Full\Path\To\Your\GT_ZP\Folder'

# Load input images
def load_images(image_dir):
    images = []
    for filename in os.listdir(image_dir):
        img = cv2.imread(os.path.join(image_dir, filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize images
            images.append(img)
    return np.array(images)

# Load masks as single-channel grayscale images
def load_masks(mask_dir):
    masks = []
    for filename in os.listdir(mask_dir):
        mask = cv2.imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)  # Load as grayscale (1 channel)
        if mask is not None:
            mask = cv2.resize(mask, (128, 128))  # Resize masks
            mask = mask[..., np.newaxis]  # Add new axis to get shape (128, 128, 1)
            masks.append(mask)
    return np.array(masks)

# Load images and masks
images = load_images(image_path)
masks_icm = load_masks(icm_mask_path)
masks_te = load_masks(te_mask_path)
masks_zp = load_masks(zp_mask_path)

# Normalization
images = images / 255.0
masks_icm = masks_icm / 255.0
masks_te = masks_te / 255.0
masks_zp = masks_zp / 255.0

# IoU metric function
def iou_metric(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return intersection / (union + K.epsilon())

# Dice Loss function
def dice_loss(y_true, y_pred):
    smooth = 1e-6  # Avoid division by zero
    intersection = K.sum(K.abs(y_true * y_pred), axis=(1, 2, 3))
    union = K.sum(y_true, axis=(1, 2, 3)) + K.sum(y_pred, axis=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

# Combo Loss (Combination of Binary Crossentropy and Dice Loss)
def combo_loss(y_true, y_pred):
    bce = losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def AtrousSpatialPyramidPooling(inputs, atrous_rates, output_channels):
    """Builds the ASPP module."""
    # 1x1 Convolution
    conv1x1 = layers.Conv2D(output_channels, kernel_size=1, padding='same', activation='relu')(inputs)

    # Atrous Convolutions
    atrous_convs = []
    for rate in atrous_rates:
        atrous_conv = layers.Conv2D(output_channels, kernel_size=3, padding='same',
                                    dilation_rate=rate, activation='relu')(inputs)
        atrous_convs.append(atrous_conv)

    # Image Pooling
    image_pooling = layers.GlobalAveragePooling2D()(inputs)
    image_pooling = layers.Reshape((1, 1, inputs.shape[-1]))(image_pooling)
    image_pooling = layers.Conv2D(output_channels, kernel_size=1, padding='same', activation='relu')(image_pooling)
    image_pooling = layers.UpSampling2D(size=(inputs.shape[1], inputs.shape[2]), interpolation='bilinear')(image_pooling)

    # Concatenate all outputs
    aspp_outputs = layers.Concatenate()([conv1x1] + atrous_convs + [image_pooling])

    # Final 1x1 Convolution
    aspp_outputs = layers.Conv2D(output_channels, kernel_size=1, padding='same', activation='relu')(aspp_outputs)

    return aspp_outputs

def DeepLabV3(input_shape=(128, 128, 3), num_classes=1, backbone='mobilenetv2', atrous_rates=(6, 12, 18)):
    """Constructs the DeepLabV3 model."""
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    backbone_output = base_model.get_layer('block_13_expand_relu').output

    # ASPP Module
    aspp = AtrousSpatialPyramidPooling(backbone_output, atrous_rates, output_channels=256)

    # Decoder
    decoder_input = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(aspp)
    decoder_output = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(decoder_input)
    
    output_icm = layers.UpSampling2D(size=(4, 4), interpolation='bilinear', name='upsampled_icm')(decoder_output)
    output_icm = layers.Conv2D(1, kernel_size=1, activation='sigmoid', name='icm_output')(output_icm)

    output_te = layers.UpSampling2D(size=(4, 4), interpolation='bilinear', name='upsampled_te')(decoder_output)
    output_te = layers.Conv2D(1, kernel_size=1, activation='sigmoid', name='te_output')(output_te)

    output_zp = layers.UpSampling2D(size=(4, 4), interpolation='bilinear', name='upsampled_zp')(decoder_output)
    output_zp = layers.Conv2D(1, kernel_size=1, activation='sigmoid', name='zp_output')(output_zp)
    
    model = Model(inputs=base_model.input, outputs=[output_icm, output_te, output_zp])
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss=combo_loss,
        metrics={
            'icm_output': [iou_metric],
            'te_output': [iou_metric],
            'zp_output': [iou_metric]
        }
    )
    return model

# Create DeepLab model with MobileNetV2 backbone
model = DeepLabV3(input_shape=(128, 128, 3), num_classes=1, backbone='mobilenetv2')
model.summary() 

# Split data into training and testing sets
X_train, X_test, y_train_icm, y_test_icm = train_test_split(images, masks_icm, test_size=0.2, random_state=42)
_, _, y_train_te, y_test_te = train_test_split(images, masks_te, test_size=0.2, random_state=42)
_, _, y_train_zp, y_test_zp = train_test_split(images, masks_zp, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, [y_train_icm, y_train_te, y_train_zp], epochs=20, batch_size=8, validation_data=(X_test, [y_test_icm, y_test_te, y_test_zp]))

# Make predictions
predicted_masks_icm, predicted_masks_te, predicted_masks_zp = model.predict(X_test)

# Output directory
output_dir = r'C:\Full\Path\To\Your\Output\Folder'
os.makedirs(output_dir, exist_ok=True)

# Display first 10 input images and predicted masks
for i in range(10):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Input image
    axes[0].imshow(X_test[i])
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    # Predicted masks
    axes[1].imshow(predicted_masks_icm[i].squeeze(), cmap='gray')
    axes[1].set_title("Predicted Mask ICM")
    axes[1].axis('off')
    
    axes[2].imshow(predicted_masks_te[i].squeeze(), cmap='gray')
    axes[2].set_title("Predicted Mask TE")
    axes[2].axis('off')
    
    axes[3].imshow(predicted_masks_zp[i].squeeze(), cmap='gray')
    axes[3].set_title("Predicted Mask ZP")
    axes[3].axis('off')
    
    # Save predicted masks to files
    cv2.imwrite(os.path.join(output_dir, f'pred_mask_icm_{i}.png'), predicted_masks_icm[i].squeeze() * 255)
    cv2.imwrite(os.path.join(output_dir, f'pred_mask_te_{i}.png'), predicted_masks_te[i].squeeze() * 255)
    cv2.imwrite(os.path.join(output_dir, f'pred_mask_zp_{i}.png'), predicted_masks_zp[i].squeeze() * 255)
    
    plt.show()

# Create IoU plots
def plot_iou_history(history, output_name):
    # IoU plots for ICM
    plt.figure(figsize=(6, 6))
    plt.plot(history.history['icm_output_iou_metric'], label='Training IoU ICM')
    plt.plot(history.history['val_icm_output_iou_metric'], label='Validation IoU ICM')
    plt.title('IoU ICM')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{output_name}_iou_ICM.png'))
    plt.show()

    # IoU plots for TE
    plt.figure(figsize=(6, 6))
    plt.plot(history.history['te_output_iou_metric'], label='Training IoU TE')
    plt.plot(history.history['val_te_output_iou_metric'], label='Validation IoU TE')
    plt.title('IoU TE')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{output_name}_iou_TE.png'))
    plt.show()

    # IoU plots for ZP
    plt.figure(figsize=(6, 6))
    plt.plot(history.history['zp_output_iou_metric'], label='Training IoU ZP')
    plt.plot(history.history['val_zp_output_iou_metric'], label='Validation IoU ZP')
    plt.title('IoU ZP')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{output_name}_iou_ZP.png'))
    plt.show()

# Call function to create IoU plots
plot_iou_history(history, 'training_history')

# Create loss plots
def plot_loss_history(history, output_name):
    # Loss plots for ICM
    plt.figure(figsize=(6, 6))
    plt.plot(history.history['icm_output_loss'], label='Training Loss')
    plt.plot(history.history['val_icm_output_loss'], label='Validation Loss')
    plt.title('Loss ICM')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{output_name}_loss_ICM.png'))
    plt.show()

    # Loss plots for TE
    plt.figure(figsize=(6, 6))
    plt.plot(history.history['te_output_loss'], label='Training Loss TE')
    plt.plot(history.history['val_te_output_loss'], label='Validation Loss TE')
    plt.title('Loss TE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{output_name}_loss_TE.png'))
    plt.show()

    # Loss plots for ZP
    plt.figure(figsize=(6, 6))
    plt.plot(history.history['zp_output_loss'], label='Training Loss ZP')
    plt.plot(history.history['val_zp_output_loss'], label='Validation Loss ZP')
    plt.title('Loss ZP')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{output_name}_loss_ZP.png'))
    plt.show()

# Call function to create loss plots
plot_loss_history(history, 'training_history')

# Display ground truth and predicted masks
def plot_predictions(X_test, y_test, y_pred, output_dir, num_samples=5):
    for i in range(num_samples):
        fig, axes = plt.subplots(2, 3, figsize=(12, 12))
        
        for j, (label, y_t, y_p) in enumerate(zip(['ICM', 'TE', 'ZP'], y_test, y_pred)):
            axes[0, j].imshow(y_t[i].squeeze(), cmap='gray')
            axes[0, j].set_title(f"Ground Truth Mask {label}")
            axes[0, j].axis('off')
            
            axes[1, j].imshow(y_p[i].squeeze(), cmap='gray')
            axes[1, j].set_title(f"Predicted Mask {label}")
            axes[1, j].axis('off')
        
        plt.tight_layout()
        plt.show()

# Call function to display ground truth and predicted masks
plot_predictions(X_test, [y_test_icm, y_test_te, y_test_zp], [predicted_masks_icm, predicted_masks_te, predicted_masks_zp], output_dir)