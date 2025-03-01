from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

def train_model(image_dir, IMG_SIZE=(128, 128), batch_size=32, epochs=10, model_save_path='face_recognition_model.h5', learning_rate=1e-4, fine_tune_from=15, reg_lambda=0.01):
    """
    Train a VGG16-based CNN model for facial recognition with fine-tuning and L2 regularization.
    
    Args:
        image_dir (str): Path to the directory containing images.
        img_size (tuple): Target size for images (height, width).
        batch_size (int): Number of samples per batch.
        epochs (int): Number of training epochs.
        model_save_path (str): Path to save the trained model.
        learning_rate (float): Learning rate for the optimizer.
        fine_tune_from (int): Layer index to start fine-tuning (use 15 for VGG16).
        reg_lambda (float): L2 regularization factor.
        
    Returns:
        model: Trained Keras model.
        history: Training history object.
        test_acc (float): Accuracy on the test set.
    """
    # Data Preparation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        image_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        image_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Load the VGG16 model with pre-trained weights, excluding the fully connected layers (top)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Freeze all layers until the `fine_tune_from` layer
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False

    # Add custom top layers for classification with Dropout and L2 regularization
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(reg_lambda))(x)  # L2 regularization
    x = Dropout(0.5)(x)  # Dropout for regularization after the dense layer
    
    output = Dense(train_generator.num_classes, activation='softmax')(x)

    # Create the complete model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model with Adam optimizer and the specified learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # First training phase: Train only the custom top layers (freeze base model)
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs  # You can adjust epochs for the first phase
    )
    
    # Fine-tuning phase: Unfreeze some of the deeper layers of the base model for fine-tuning
    for layer in base_model.layers[fine_tune_from:]:
        layer.trainable = True  # Unfreeze layers for fine-tuning

    # Recompile the model after unfreezing some layers
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Fine-tuning phase: Continue training with the whole model (base + top layers)
    history_finetune = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs  # You can adjust epochs for the fine-tuning phase
    )
    
    # Save the model after fine-tuning
    model.save(model_save_path)
    
    # Evaluate the model on the test set
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        image_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc:.2f}")
    
    return test_acc
