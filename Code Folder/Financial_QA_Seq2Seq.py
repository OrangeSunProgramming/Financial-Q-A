#We are going to use the small pre-trained t5 model from google to fine tune it on our
#financial QA dataset on "question" and "answer" columns which contains financial related questions and answers

#Importing all the library we need for this project
import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM #Importing the model from the huggingface platform
from datasets import Dataset
import os

#Loading the dataset using the pandas library
financial_dataset = pd.read_csv("/content/drive/MyDrive/QA/Financial_QA/Financial_QA.csv")

#Loading the model google/flan-t5-small and the respective tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

#As a start I am freezing all encoder layers except the last three layers
#I want the model to retain some pre-trained knowledge weight while
#Adapting to my dataset
for layer in model.encoder.block[:-3]:
  layer.trainable = False

#Unfreezing all the decoder layers in the model to fine tune it to my datasets
#I want the model to be able to generate responses using my financial dataset
for layer in model.decoder.block:
  layer.trainable = True

#Looking at the model's summary
model.summary()

#Now I am going to prepare the dataset and preprocess it to be able to pass it through
#the model and fine tune it.

#The "question" and "answer" column in the financial_dataset contain numerical values as well
#We need to turn it into string format to pass it to the model for fine tuning

#We are now cleaning the dataset by turning every value in string format
financial_dataset_cleaned = {
    "question": [str(question) for question in financial_dataset["question"]],
    "answer": [str(answer) for answer in financial_dataset["answer"]]
}

#turning the cleaned financial dataset into a Dataset
dataset = Dataset.from_dict(financial_dataset_cleaned)

#We need to tokenize the dataset. We are padding to the longest string in the dataset
tokenized_question = tokenizer(dataset["question"], padding=True, truncation=True, return_tensors="tf")
tokenized_answer = tokenizer(dataset["answer"], padding=True, truncation=True, return_tensors="tf")

#Preparing TensorFlow Dataset. We are a using a TensorFlow Dataset dictionary with the respective key needed for fine tuning.
#This step is crucial since we are using TFAutoModelForSeq2SeqLM
train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_ids": tokenized_question["input_ids"],
        "attention_mask": tokenized_question["attention_mask"],
        "decoder_input_ids": tokenized_answer["input_ids"],
        "decoder_attention_mask": tokenized_answer["attention_mask"]
    },
    tokenized_answer["input_ids"]
))

#We are going to shuffle the training TensorFlow Dataset with a buffer size the length of the dataset
train_dataset = train_dataset.shuffle(buffer_size=len(dataset))

#Splitting the training dataset into a portion for training and another for validating
validation_split = 0.2 #Splitting 80% for training and 20% for validation
validation_size = int(len(dataset)*validation_split)

#Creating the training dataset and validation dataset with their respective split
training_dataset = train_dataset.take(validation_size)
validation_dataset = train_dataset.skip(validation_size) #Makes sure that the validation training set does not overlap with the training dataset

#Batching the dataset for fine tuning. We are going to use a batch size of 16 since we have a T4 GPU.
batch_size = 16 #Since we have a training_dataset of 5600 data, then there will be 350 iterations in training per epoch.
train_dataset = train_dataset.batch(batch_size)
validation_dataset = validation_dataset.batch(batch_size)

#We are now going to set up the model for fine tuning. Since we have a T4 GPU available,
#we are going to use an early stopping function that monitors the validation accuracy
#we are also going to create our model checkpoint function that monitors the validation loss
#keep in mind that tf.keras.callbacks.ModelCheckpoint() expects our model to be sequential or functional
#applying the early stopping callback prevents overfitting and the custom model checkpoint saves the best weights and biases only with respect to the validation loss
#which is essential in case the model stops for any reason and we need to keep training

#EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5)

# Custom model checkpoint callback
class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, model, filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=0):
        super(CustomModelCheckpoint, self).__init__()
        self.model = model
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_value = float('inf') if mode == 'min' else float('-inf')

        # Ensure directory exists
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.monitor)
        if current_value is None:
            if self.verbose > 0:
                print(f"Metric '{self.monitor}' not found in logs. Skipping checkpoint.")
            return

        if (self.mode == 'min' and current_value < self.best_value) or (self.mode == 'max' and current_value > self.best_value):
            self.best_value = current_value
            if self.verbose > 0:
                print(f"** Best Model Saved! Checkpoint: Epoch {epoch + 1} - {self.monitor}: {current_value:.4f}")
            self.model.save_weights(self.filepath.format(epoch=epoch + 1, **logs))

# Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))

# Define the checkpoint callback
checkpoint_callback = CustomModelCheckpoint(model,
                                            filepath="drive/MyDrive/QA/Financial_QA/google_flan_t5_small_june_29_day_time_2_02_pm_best_model_weights_biases.h5",
                                            monitor="val_loss",
                                            save_best_only=True)

# Fitting the model
num_epochs = 50
model_history = model.fit(train_dataset,
                          epochs=num_epochs,
                          validation_data=validation_dataset,
                          callbacks=[early_stopping, checkpoint_callback])

# model.load_weights(checkpoint_callback.filepath)