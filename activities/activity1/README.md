# Lab Instructions: Interacting with a Local OpenAI API Server Using TinyLLM

## Objective

This lab will guide you through setting up and using a local OpenAI API server with the TinyLLM model. You will learn how to generate text completions, create embeddings, and format responses in Markdown.

## Prerequisites

- Python environment with the OpenAI library installed.
- Access to a local OpenAI API server running at `http://127.0.0.1:8080/v1`. 
  - For this activity we will be using **llamafile** 
- Basic understanding of Python programming.

## Step-by-Step Instructions

## Step 1: Import Required Libraries

Begin by importing the necessary libraries for interacting with the OpenAI API and displaying Markdown:

```python
from openai import OpenAI
from IPython.display import Markdown, display
```

## Step 2: Set Up the OpenAI Client

Configure the OpenAI client to connect to your local API server:

```python
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",  # Replace with your API server's IP and port if different
    api_key="sk-no-key-required"
)
```

- **base_url**: The URL where your local API server is running.
- **api_key**: No key is required for this local setup.

## Step 3: Generate Text Completions

Use the client to generate text completions with different parameters:

```python
completion = client.chat.completions.create(
  model="TinyLLM",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is a Large Language Model?"}
  ]
)

print(completion.choices[0].message.content)
```

- **model**: Specifies the model to use, in this case, `TinyLLM`.
- **messages**: A list of messages that simulate a conversation, including system and user roles.
- **temperature** (optional): Controls the randomness of the output.
- **max_tokens** (optional): Limits the number of tokens in the response.

## Step 4: Create Embeddings

1. **Creating Embeddings**: The code uses the `client.embeddings.create` method to generate embeddings for the two text inputs. This process involves converting the text into vector representations that capture their semantic meanings.
2. **Comparing Embeddings**: The activity compares the embeddings of the two phrases to assess their semantic similarity. This is done by examining the vector representations and calculating the distance or similarity metric between them.
3. **Understanding Similarity**: The comparison helps understand how similar the two phrases are in terms of their semantic meaning. This is essential for NLP tasks that rely on understanding the relationships between words and phrases.

Generate embeddings for text inputs to compare semantic similarity:

```python
q1 = client.embeddings.create(
    model='TinyLLama',
    input="What is generative AI?"
)

q2 = client.embeddings.create(
    model='TinyLLama',
    input="What is a Large Language Model?"
)

q1 = q1.data[0].embedding
q2 = q2.data[0].embedding

print(q1[0:10])
print(q2[0:10])
```

- **embeddings.create**: Generates vector representations for given text inputs.
- Compare embeddings to assess similarity between different phrases.

You can measure similarities between statements, for example two vectors using Cosine Similarity. Create this function:

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (magnitude_vec1 * magnitude_vec2)
    print(f"Cosine Similarity: {similarity}")
```

Here is an example output using the function:

```python
cosine_similarity(q1, q2)
```

You will get the following:

```
Cosine Similarity: 0.8884938882374027
```

This indicates a high degree of similarity. 

Cosine similarity values range from -1 to 1:
* `1`: The vectors are identical in direction.
* `0`: The vectors are orthogonal (i.e., they have no similarity).
* `-1`: The vectors are diametrically opposed.

Try another example:

```python
```python
q1 = client.embeddings.create(
    model='Llama',
    input="The dog was running all over"
)

q2 = client.embeddings.create(
    model='Llama',
    input="The weather was horrible"
)

q1 = q1.data[0].embedding
q2 = q2.data[0].embedding
```

Here is what I got for Cosine Similarity

```
Cosine Similarity: 0.5674563369286211
```

What does that tell us?

## What are Embeddings?

Embeddings are vector representations of words or phrases in a high-dimensional space.They capture the semantic and syntactic relationships between words, allowing machines to understand and process language more effectivel

## Purpose of the Activity

The activity aims to demonstrate how embeddings can be used to compare the semantic similarity between two text inputs. By creating embeddings for two different phrases, "What is Generative AI?" and "What is a Large Language Model?", the activity illustrates how these vector representations can be compared to assess their similarity


## Step 5: Format Responses in Markdown

Generate and display responses formatted in Markdown:

```python
completion = client.chat.completions.create(
    model="TinyLLM",
    messages=[
        {"role": "system", "content": "You are a helpful assistant.Respond in markdown"},
        {
            "role": "user",
            "content": "Write an example of a Hello World in Python with explanation."
        }
    ]
)
print(completion.choices[0].message.content)
display(Markdown(completion.choices[0].message.content))
```

- **Respond in markdown**: Instructs the assistant to format its response using Markdown syntax.
- **display(Markdown(...))**: Renders the Markdown content for better visualization.

## Step 6: Define a Function for Reusable Queries

Create a function to streamline querying the API with different prompts:

```python
def responses(user_prompt):
    completion = client.chat.completions.create(
        model="TinyLLM",
        messages=[
            {"role": "system", "content": "You are a helpful assistant.Respond in markdown"},
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )
    return completion.choices[0].message.content

response = responses("Write me a Hello World example in Java with explanations and steps in bullet points")
display(Markdown(response))
```

- **responses function**: Takes a user prompt and returns a formatted response from the API.
- Use this function to easily generate responses for various queries.

## Conclusion

By following these steps, you can interact with a local OpenAI API server using TinyLLM to generate text completions, create embeddings, and format responses in Markdown. This lab demonstrates how to leverage AI models for natural language processing tasks effectively.

## Challenge

Try a new llamafile, this time the impressive **LLaMA 3.2 1B Instruct**

This is a large language model that was released by Meta on `2024-09-25`. This edition of LLaMA packs a lot of quality in a size small enough to run on most computers with 4GB+ of RAM

To download run:
```bash
wget https://huggingface.co/Mozilla/Llama-3.2-1B-Instruct-llamafile/resolve/main/Llama-3.2-1B-Instruct.Q6_K.llamafile
```

After downloading you need to add premission:
```bash
chmod +x Llama-3.2-1B-Instruct.Q6_K.llamafile
```

Finally, run as a server

```bash
./Llama-3.2-1B-Instruct.Q6_K.llamafile --server
```

Rerun the previous code and comapre the outputs using the new model