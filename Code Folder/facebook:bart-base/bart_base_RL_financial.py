#Importing Libraries
import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from bert_score import score as bert_score
import sacrebleu
import pandas as pd
import numpy as np

#Loading Model and Tokenizer
model_name = "facebook/bart-base"
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#Loading Dataset
data = pd.read_csv('/content/drive/MyDrive/QA/Financial_QA/Financial_QA.csv')


data = data.iloc[:250] #Considering the first 250 data points

cleaned_data = {
    "question": [str(question) for question in data["question"]],
    "answer": [str(answer) for answer in data["answer"]]
} #makes sure that everything passes as a string

questions = data["question"]
answers = data["answer"]

#Defining Reward Function
def calculate_reward(prediction, reference):
    #Coherence using BERTScore
    P, R, F1 = bert_score([prediction], [reference], lang='en', model_type='bert-base-uncased')
    coherence_score = F1.mean().item()

    #Fluency using BLEU
    fluency_score = sacrebleu.corpus_bleu([prediction], [[reference]]).score / 100  # Normalize BLEU to be between 0 and 1

    #Defining thresholds and weights
    coherence_threshold = 0.85  
    coherence_penalty = 0.9    
    fluency_weight = 0.2
    coherence_weight = 0.8 #coherence is more important than fluency in this case

    #Calculating final reward
    if coherence_score < coherence_threshold:
        reward = coherence_weight * (coherence_score - coherence_penalty) + fluency_weight * fluency_score
    else:
        reward = coherence_weight * coherence_score + fluency_weight * fluency_score

    return reward, coherence_score, fluency_score

#Training Loop
epochs = 7
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
checkpoint_dir = '/content/drive/MyDrive/QA/Financial_QA/bart_base_RL_financial_dataset_model_folder/Bertscore8_RL_training_code_folder'
checkpoint_prefix = checkpoint_dir + "/ckpt"
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

#Restoring the latest checkpoint
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print(f"Restored from {latest_checkpoint}")
else:
    print("Starting training from scratch.")

for epoch in range(epochs):
    total_loss = 0
    total_coherence_score = 0
    total_fluency_score = 0
    for question, reference in zip(questions, answers):
        #Tokenize inputs and labels
        input_ids = tokenizer(question, return_tensors='tf', padding='max_length', truncation=True, max_length=250).input_ids
        target_ids = tokenizer(reference, return_tensors='tf', padding='max_length', truncation=True, max_length=250).input_ids

        with tf.GradientTape() as tape:
            outputs = model(input_ids, labels=target_ids)
            logits = outputs.logits
            loss = outputs.loss

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        prediction_ids = tf.argmax(logits, axis=-1)
        prediction = tokenizer.decode(prediction_ids[0], skip_special_tokens=True)

        reward, coherence_score, bleu = calculate_reward(prediction, reference)
        total_loss += loss.numpy() - reward
        total_coherence_score += coherence_score
        total_fluency_score += bleu

    avg_loss = total_loss / len(questions)
    avg_coherence_score = total_coherence_score / len(questions)
    avg_fluency_score = total_fluency_score / len(questions)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, Coherence Score: {avg_coherence_score}, Fluency Score: {avg_fluency_score}")

    #Saving checkpoint
    checkpoint.save(file_prefix=checkpoint_prefix)

#Saving the Model
model.save_pretrained(checkpoint_dir)
tokenizer.save_pretrained(checkpoint_dir)