from llama_cpp import Llama  


# Load the GGUF model from the local file
model_path = "mistral-7b-v0.1.Q3_K_S.gguf"  # Ensure the correct path

# Explicitly set the max_seq_len

# Load TinyLlama from the local file instead of Hugging Face
llm = Llama(model_path="/Users/mahbub/recipe_agent/models/mistral-7b-v0.1.Q3_K_S.gguf", n_ctx=2048) 

def generate_human_friendly_response(matched_texts):
    prompt = f"<|system|>You are an AI assistant helping with document retrieval. Provide a clear, concise answer.</s>\n"
    prompt += f"<|user|>Given the following text snippets, summarize and answer in a human-friendly way:\n\n"
    
    for idx, text in enumerate(matched_texts, 1):
        prompt += f"Text {idx}: {text}\n"
    
    prompt += "</s>\n<|assistant|>"

    # Generate response
    response = llm(prompt, max_tokens=200)
    
    # Extract and return the response
    return response["choices"][0]["text"].strip() if "choices" in response else ""

