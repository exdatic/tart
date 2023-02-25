import numpy as np
import sys
import torch
import torch.nn.functional as F
from transformers import pipeline

sys.path.append('tart/TART')

from src.modeling_enc_t5 import EncT5ForSequenceClassification
from src.tokenization_enc_t5 import EncT5Tokenizer


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global tokenizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load TART full and tokenizer
    model = EncT5ForSequenceClassification.from_pretrained("facebook/tart-full-flan-t5-xl").to(device)
    tokenizer = EncT5Tokenizer.from_pretrained("facebook/tart-full-flan-t5-xl")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer
    # Parse out your arguments
    pairs = model_inputs.get('sentence_pairs')
    
    instr = "Retrieve a scientific paper sentence that answers this question"
    queries = [f"{instr} [SEP] {p[0]}" for p in pairs]
    sents = [p[1] for p in pairs]
    features = tokenizer(queries, sents, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(**features).logits
        normalized_scores = [float(score[1]) for score in F.softmax(scores, dim=1)]
    # Return the results as a dictionary
    return {"output": normalized_scores}
    return result
