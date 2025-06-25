# template.py

def get_template(model_template):
    """
    Returns the appropriate template function based on the model type.
    """
    templates = {
        "qwen": qwen_template,
        "gemma": gemma_template,
        # Add more templates here as needed
    }
    return templates.get(model_template, qwen_template)  # Default to qwen_template if not found

def qwen_template(prompt, assistant_response, tokenizer):
    """
    Template for Qwen model.
    """
    return prompt + assistant_response + tokenizer.eos_token

def gemma_template(prompt, assistant_response, tokenizer):
    """
    Template for Gemma model.
    """
    # print("Gemma template")
    # print("prompt: ", prompt)
    # print("assistant_response: ", assistant_response)
    # print("tokenizer.eos_token", tokenizer.eos_token)    
    return prompt + assistant_response + "<end_of_turn>" + tokenizer.eos_token

# Add more template functions here as needed
