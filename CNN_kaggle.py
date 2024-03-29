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
EPOCHS = 1

# Get current time
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class DetailedLoggingCallback(Callback):
    def __init__(self, test_data, file_prefix="CNN_Kaggle_optAdam_lr0.001_bs32"):
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
        y_true_test = []
        y_pred_test = []
        for images, labels in self.test_data:
            y_true_test.extend(np.argmax(labels.numpy(), axis=1))
            y_pred_test.extend(np.argmax(self.model.predict(images), axis=1))

        cm_test = confusion_matrix(y_true_test, y_pred_test)
        report_test = classification_report(
            y_true_test, y_pred_test, digits=5, output_dict=True
        )
        precision_test = precision_score(y_true_test, y_pred_test, average="macro")
        recall_test = recall_score(y_true_test, y_pred_test, average="macro")
        f1_test = f1_score(y_true_test, y_pred_test, average="macro")
        mcc_test = matthews_corrcoef(y_true_test, y_pred_test)
        cmc_test = cohen_kappa_score(y_true_test, y_pred_test)

        print("Confusion Matrix (Test):")
        print(cm_test)
        print("Classification Report (Test):")
        print(report_test)

        self.epoch_cm_logs.append((epoch + 1, cm_test))
        self.epoch_report.append((epoch + 1, report_test))
        self.epoch_logs.append(
            (
                epoch + 1,
                logs.get("loss", 0),
                logs.get("accuracy", 0),
                logs.get("val_loss", 0),
                logs.get("val_accuracy", 0),
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

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", input_shape=(224, 224, 3)),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomHeight(0.2),
        tf.keras.layers.RandomWidth(0.2),
        tf.keras.layers.Rescaling(1.0 / 255),
    ],
    name="data_augmentation",
)
train_data = (
    tf.keras.preprocessing.image_dataset_from_directory(
        directory=train_dir,
        image_size=IMG_SIZE,
        label_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    .cache()
    .shuffle(1000)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_data = (
    tf.keras.preprocessing.image_dataset_from_directory(
        directory=test_dir,
        image_size=IMG_SIZE,
        label_mode="categorical",
        batch_size=BATCH_SIZE,
    )
    .cache()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

detailed_logging_callback = DetailedLoggingCallback(test_data=test_data)
input_shape = (32, 224, 224, 3)
model = tf.keras.models.Sequential(
    [
        data_augmentation,
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, activation="relu", input_shape=input_shape
        ),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2, padding="valid"),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)
model.build(input_shape=input_shape)

model.summary()


model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data,
    validation_steps=len(test_data),
    callbacks=[
        detailed_logging_callback,
    ],
)


model.save("./CNN_kaggle.keras")
