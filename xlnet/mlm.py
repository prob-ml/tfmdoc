from transformers import XLNetTokenizer, XLNetModel, XLNetForSequenceClassification, XLNetForMultipleChoice
import torch

tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
model = XLNetModel.from_pretrained('xlnet-large-cased')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=False)).unsqueeze(0)  # Batch size 1

outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
