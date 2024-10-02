# **Instructions for the Lab**

**Objective**: In this lab, you will create a simple email writing assistant using OpenAI's API. The assistant will take input from the user, such as the subject of the email, recipient's name, and additional information, and generate a professional email using a large language model.

### Code Template

Below is the code template with key areas left as placeholders. Your goal is to complete these sections.

```python
import openai

client = OpenAI(
    base_url="", 
    api_key = ""
)

# Function to generate an email
def generate_email(subject, recipient_name, additional_info):
    # Step 1: Create the prompt to guide the AI in generating the email
    prompt = f"Write a professional email to {recipient_name} with the subject '{subject}'. Include the following information: {additional_info}"
    
    # Step 2: Call the OpenAI API to generate the email
    # Hint: Use the 'chat' endpoint to create a completion based on the prompt
    response = client.__________.create(
        model="TinyLLM",  # Replace "TinyLLama" with the correct model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=  __,# add a value,
        temperature= __,# add a value,
        top_p= __,# add a value,
        frequency_penalty= __,# add a value,
        presence_penalty= __,# add a value
    )
    
    # Step 3: Extract and return the generated email content from the API response
    return response.__________[0].__________.__________  # <-- Complete this line to get the email content

# Function to start the email writing assistant
def email_writing_assistant():
    print("Welcome to the Email Writing Assistant!\n")
    
    # Step 4: Gather user input for the email subject, recipient name, and additional information
    subject = input("Enter the email subject: ")
    recipient_name = input("Enter the recipient's name: ")
    additional_info = input("Enter any additional information to include in the email: ")
    
    # Step 5: Call the 'generate_email' function and display the generated email
    email = generate_email(subject, recipient_name, additional_info)
    print(f"Email:\n{email}")

# Start the email writing assistant
email_writing_assistant()

```

## OpenAI API Parameters

### **Max Tokens**

- **Purpose:** `max_tokens` determines the maximum number of tokens (words andcharacters) that can be generated in the chat completion. This parameter helpscontrol the verbosity of the response.
- **Range:** Integer, typically between 0 and 2048, though it can vary depending on the model and context.
- **Usage:** Lower values (e.g., 50-100) are used for concise summaries, while highervalues are used for longer, more detailed responses

### **Temperature**

- **Purpose:** `temperature` controls the randomness of the output. It influences how creative or deterministic the responses are.
- **Range:** Number between 0 and 1.
  - **0:** Generates the most deterministic and predictable response.
  - **1:** Generates the most random and creative response.
- **Usage:** Lower values (e.g., 0.2) make the output more deterministic and focused, while higher values (e.g., 0.7) make it more random and creative. 
- **Default Value**: 0.8

### **Top P (Nucleus Sampling)**

- **Purpose:** `top_p` dictates the variety in responses by only considering the top ‘P’ percent of probable words. It is an alternative to sampling with temperature. In other words, it controls the diversity of the generated text.
- **Range:** Number between 0 and 1.
- **Usage:** Lower values (e.g., 0.8) make responses more predictable, while higher values increase diversity and surprise. For example, a value of 0.1 means only the top 10% of probable words are considered
- **Default Value**: 0.95

Example of finding the balance between **Temperature** and **Top P** Parameters 

<img src="images/649f076eeeff5a9fa30e442a_OpenAI Temperature Recomendations.png" alt="Table with use cases and corresponding Temperate and Top_ p values from OpenAI" style="zoom:50%;" />

### **Frequency Penalty**

- **Purpose:** `frequency_penalty` reduces repetition by decreasing the likelihood off frequently used words. It penalizes new tokens based on their existing frequency in the text so far.
- **Range:** Number between -2.0 and 2.0.
- **Usage:** Positive values penalize repetition, making the model less likely to repeat the same line verbatim. For example, a value of 0.5 would moderately reduce repetition
- **Default Value** is `0` (disabled)

### **Presence Penalty**

- **Purpose:** `presence_penalty` promotes the introduction of new topics in the conversation. It penalizes new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
- **Range:** Number between -2.0 and 2.0.
- **Usage:** Positive values encourage diverse ideas and minimize repetition. For example, a value of 0.5 would moderately encourage new topics
- **Default Value** is `0` (disabled)

These parameters are crucial for fine-tuning the. For more information on the other parameters and their default values you can read here:

* [Llamafile Server EndPoint Documentation](https://github.com/Mozilla-Ocho/llamafile/blob/main/llama.cpp/server/README.md#api-endpoints)
* [OpenAI Chat Completion Documentation](https://platform.openai.com/docs/api-reference/chat/create)

### Prompt Bonus Challenge

Modify the template to include different email tones (e.g., formal, friendly, urgent). Add an extra input where the user can select a tone, and adjust the prompt accordingly.