
import os
import onnxruntime as ort
import numpy as np
from transformers import GPT2Tokenizer
from kivy.utils import platform

def load_model():
    if platform == 'android':
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, '..', 'models', 'gpt2_model.onnx')
    else:
        model_path = os.path.join('models', 'gpt2_model.onnx')

    session = ort.InferenceSession(model_path)
    return session

def load_tokenizer():
    return GPT2Tokenizer.from_pretrained('gpt2')

def preprocess_text(text, tokenizer, max_length=1024):
    return tokenizer.encode(text, return_tensors='np', max_length=max_length, truncation=True)

def run_inference(session, input_ids):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_ids})
    return outputs[0]

def generate_text(session, tokenizer, prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
    input_ids = preprocess_text(prompt, tokenizer)
    generated = list(input_ids[0])

    for _ in range(max_length):
        inputs = np.array([generated]).astype(np.int64)
        outputs = run_inference(session, inputs)
        next_token_logits = outputs[0, -1, :] / temperature
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = softmax(filtered_logits)
        next_token = np.random.choice(len(probs), p=probs)
        generated.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # Implementation remains the same as in the previous example
    # Add filtering logic here
    if top_k > 0:
        indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_indices = np.argsort(logits)[::-1]
        cumulative_probs = np.cumsum(softmax(logits[sorted_indices]))
        indices_to_remove = sorted_indices[cumulative_probs > top_p]
        logits[indices_to_remove] = filter_value

    return logits

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()