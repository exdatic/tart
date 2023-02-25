# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
import sys
sys.path.append('tart/TART')

from src.modeling_enc_t5 import EncT5ForSequenceClassification
from src.tokenization_enc_t5 import EncT5Tokenizer




def download_model():
    # load TART full and tokenizer
    model = EncT5ForSequenceClassification.from_pretrained("facebook/tart-full-flan-t5-xl")
    tokenizer = EncT5Tokenizer.from_pretrained("facebook/tart-full-flan-t5-xl")

if __name__ == "__main__":
    download_model()
