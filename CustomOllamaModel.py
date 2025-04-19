import ollama

modelfile = """
FROM llama3.2
SYSTEM Your are a very Smart assistant who answers questions succintly and informatively with friendy tone.
PARAMETER temperature 0.1
"""

#Create custom ollama model with name "dost" from "llama3.2"
ollama.create(model="dost", from_="llama3.2",system=modelfile)

response = ollama.chat( 
    model="dost",
    messages=[
        {"role":"user", "content":"what is India, tell me in consice format?"},
    ],
    stream=True,
)


for chunk in response:
    print(chunk["message"]["content"], end="", flush=True)