import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Qwen 2.5-3B model and tokenizer from Hugging Face
model_name = "Qwen/qwen2.5-3b"  # Correct model name for Qwen 2.5-3B
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to generate response using Qwen model
def get_ai_response(prompt):
    # Tokenize the input
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate a response from the model
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    
    # Decode the response and return it
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.title("AI Chatbot with Qwen 2.5-3B")

# Create a chat history (persistent memory)
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Get user input
user_input = st.text_input("You:", "")

if user_input:
    # Add the user input to the chat history
    st.session_state['messages'].append({"role": "user", "content": user_input})

    # Generate AI response
    ai_response = get_ai_response(user_input)
    
    # Add AI response to the chat history
    st.session_state['messages'].append({"role": "ai", "content": ai_response})

# Display the chat history
for message in st.session_state['messages']:
    if message['role'] == 'user':
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**AI:** {message['content']}")
