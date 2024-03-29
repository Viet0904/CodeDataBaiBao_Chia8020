import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import datetime
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
)
from tensorflow.keras.callbacks import Callback

# Define necessary parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 100

# Get current time
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class DetailedLoggingCallback(Callback):
    def __init__(self, test_data, file_prefix="CNN_optAdam_lr0.001_bs32"):
        super(DetailedLoggingCallback, self).__init__()
        self.test_data = test_data
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.detail_file_path = f"{file_prefix}_{current_time}_details.txt"
        self.Confusion_Matrix_path = f"{file_prefix}_{current_time}_confusion_matrix"
        self.report_path = f"{file_prefix}_{current_time}_report"
        # Initialize file and write header with tab as separator
        with open(self.detail_file_path, "w") as f:
            f.write(
                "Epoch\tTrain Loss\tTrain Accuracy\tTest Loss\tTest Accuracy\tTest Precision\tTest Recall\tTest F1-Score\tTest MCC\tTest CMC\n"
            )
        with open(f"{self.Confusion_Matrix_path}_test.txt", "w") as f:
            f.write("Confusion Matrix Test\n")
        with open(f"{self.report_path}_test.txt", "w") as f:
            f.write("Classification Report Test\n")
        self.epoch_logs = []
        self.epoch_cm_logs = []
        self.epoch_report = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        test_results = self.model.evaluate(self.test_data, verbose=0)
        test_loss, test_accuracy = test_results[0], test_results[1]
        y_pred_test = np.argmax(self.model.predict(self.test_data), axis=1)
        y_true_test = self.test_data.classes
        cm_test = confusion_matrix(y_true_test, y_pred_test)
        report_test = classification_report(
            y_true_test, y_pred_test, digits=5, output_dict=True
        )
        precision_test = precision_score(y_true_test, y_pred_test, average="macro")
        recall_test = recall_score(y_true_test, y_pred_test, average="macro")
        f1_test = f1_score(y_true_test, y_pred_test, average="macro")
        mcc_test = matthews_corrcoef(y_true_test, y_pred_test)
        cmc_test = cohen_kappa_score(y_true_test, y_pred_test)
        test_accuracy = logs.get("accuracy", 0)
        cm_test = confusion_matrix(y_true_test, y_pred_test)
        report_test = classification_report(
            y_true_test, y_pred_test, digits=5, output_dict=True
        )
        print("Confusion Matrix (Test):")
        print(cm_test)
        print("Classification Report (Test):")
        print(report_test)
        self.epoch_cm_logs.append((epoch + 1, cm_test))
        self.epoch_report.append((epoch + 1, report_test))
        # Save information to temporary list with values separated by tab
        self.epoch_logs.append(
            (
                epoch + 1,
                logs.get("loss", 0),
                logs.get("accuracy", 0),
                test_loss,
                test_accuracy,
                precision_test,
                recall_test,
                f1_test,
                mcc_test,
                cmc_test,
            )
        )

    def on_train_end(self, logs=None):
        # Save information from each epoch to detail file, using tab as separator
        with open(self.detail_file_path, "a") as f:
            for log in self.epoch_logs:
                f.write(
                    f"{log[0]}\t{log[1]:.5f}\t{log[2]:.5f}\t{log[3]:.5f}\t{log[4]:.5f}\t{log[5]:.5f}\t{log[6]:.5f}\t{log[7]:.5f}\t{log[8]:.5f}\t{log[9]:.5f}\n"
                )
        with open(f"{self.Confusion_Matrix_path}_test.txt", "a") as f:
            for log in self.epoch_cm_logs:
                f.write(f"{log[1]}\n\n")
        with open(f"{self.report_path}_test.txt", "a") as f:
            for log in self.epoch_report:
                f.write(f"{log[1]}\n\n")



# Create paths to data directories
train_dir = "./Guava_Dataset/Train"
test_dir = "./Guava_Dataset/Test"

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    horizontal_flip=False,
    fill_mode="nearest",
)

# Load data
train_data = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
# Đánh giá mô hình trên tập test
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load tập test
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
detailed_logging_callback = DetailedLoggingCallback(
    test_data=test_data
)
# Create a CNN model
model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

# Compile the model with new metrics
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
)

# Train the model with the new callback
history = model.fit(
    train_data,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[detailed_logging_callback],
)

# Save the model in the native Keras format
model.save("./CNN.keras")


