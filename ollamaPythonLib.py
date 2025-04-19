#Importing ollama library
import ollama 

# == Chat Example
response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role":"user", "content":"what is India?"},
    ],
    stream=True,
)


for chunk in response:
    print(chunk["message"]["content"], end="", flush=True)


