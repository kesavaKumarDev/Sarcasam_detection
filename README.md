**Sarcasm Detection using Transformers**

This project implements sarcasm detection using state-of-the-art Transformer models from Hugging Face's Transformers library. The models used include BERT, MobileBERT, DistilBERT, RoBERTa, and GPT-2. The goal is to classify whether a given text is sarcastic or not, leveraging pre-trained models fine-tuned on the Kaggle dataset "Sarcasm Corpus v2" by Oraby et al.

**Dataset**

The dataset used for training and evaluation is the "Sarcasm Corpus v2" by Oraby et al., available on Kaggle. It contains labeled examples of sarcastic and non-sarcastic texts, providing a robust foundation for training sarcasm detection models.

**Models Used**

BERT: Bidirectional Encoder Representations from Transformers.
MobileBERT: Efficient variant of BERT optimized for mobile and edge devices.
DistilBERT: Lighter and faster version of BERT.
RoBERTa: Robustly optimized BERT approach.
GPT-2: OpenAI's transformer model for text generation and understanding.

**Implementation Details**

Preprocessing: Text preprocessing includes tokenization, padding, and attention masking using the Hugging Face tokenizers library.
Model Training: Models are fine-tuned using the Transformers library's training utilities, optimizing for sarcasm classification.
Evaluation: Performance metrics such as accuracy, precision, recall, and F1 score are computed to assess model effectiveness.
Setup and Usage
Environment Setup: Install required libraries using requirements.txt.

**Steps:**

**Step-1**: **pip install -r requirements.txt**
**Step-2**: Fine-tune the selected model(s) on the dataset.
   **python train.py --model_name bert-base-uncased --epochs 3 --batch_size 32**
**Step-3**: Use the trained model to predict sarcasm in new texts.
    **python predict.py --model_name bert-base-uncased --text "This is clearly the best day ever..."**
**Step-4**: Evaluation: Evaluate model performance on a validation set.
     **python evaluate.py --model_name bert-base-uncased --val_data val.csv**

**Future Improvements**

Experiment with different model architectures and hyperparameters.
Incorporate ensemble methods for improved prediction accuracy.
Explore additional data augmentation techniques for better generalization.
