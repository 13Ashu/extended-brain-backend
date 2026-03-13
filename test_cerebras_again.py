from openai import OpenAI

# Replace with your Cerebras API key
API_KEY = ""

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.cerebras.ai/v1"
)

response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain transformers in 3 simple sentences."}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)