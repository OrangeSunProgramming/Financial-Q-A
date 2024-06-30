# Financial-Q-A

## Financial Question-Answering with Google Flan T5 Models

This project aims to develop a question-answering model using FLAN-T5 models such as google/flan-t5-small, google/flan-t5-base, and google/flan-t5-large from HuggingFace. The model is trained on a financial dataset obtained from Kaggle under the Apache License 2.0, consisting of 7000 rows and 5 columns, with focus on the "question" and "answer" fields.

To leverage the model's pre-trained weights effectively while adapting to the financial domain, a strategy was employed where certain encoder layers were frozen while all decoder layers were unfrozen. Initial experiments with unfreezing all layers showed diminished performance, highlighting the need for a balanced approach. Specifically, training involved 3 epochs resulting in metrics (loss: 0.0680, accuracy: 0.9845, val_loss: 0.0075, val_accuracy: 0.9983), demonstrating challenges in generalization due to dataset constraints.

### Data Preparation and Training

The dataset underwent preprocessing to accommodate numerical values and facilitate input into the model. Using tools from the `transformers` library in Python, data cleaning and tokenization were performed. The dataset was structured into a TensorFlow Dataset format with features including "input_ids" and "attention_mask". 

Given the dataset's size (5600 for training and 1400 for validation), strategies to mitigate overfitting were crucial. An early stopping callback was implemented to monitor validation accuracy, coupled with a custom model checkpoint to save the best weights based on validation loss. This approach provided flexibility not achievable with standard TensorFlow callbacks, ensuring optimal training outcomes.

### Model Configuration and Training

The model was compiled using the Adam optimizer with learning rates (1e-5, 3e-5, 2e-3) and utilized SparseCategoricalCrossentropy(from_logits=True) as the loss function. Training spanned 50 epochs to achieve convergence, aiming to optimize the sequence-to-sequence model's performance in a challenging, small-scale financial question-answering dataset.

### Results and Conclusion

The primary objective was to fine-tune large-scale models effectively on limited data, achieving high accuracy in generating text responses within the financial domain. Detailed results can be found in the Results folder of this repository.

---

### Licensing

This project is licensed under the MIT License (see LICENSE file for details). While the MIT License permits commercial use, we encourage non-commercial applications. We appreciate attribution if this codebase is used or modified (refer to the NOTICE.txt file for attribution guidelines).