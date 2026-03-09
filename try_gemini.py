import os
from google import genai

# The client automatically picks up the API key from the environment variable GEMINI_API_KEY
# Alternatively, you can pass the key explicitly: client = genai.Client(api_key="YOUR_API_KEY")
client = genai.Client(api_key="")

# Select a model, for example, 'gemini-2.5-flash'
# model_name = "gemini-2.5-flash"
model_name = "gemini-3.1-pro-preview"

# Define your prompt
prompt = "Explain how AI works in a few words"

# Generate content
response = client.models.generate_content(
    model=model_name,
    contents=prompt
)

print(response)

# Print the response text
print(response.text)
