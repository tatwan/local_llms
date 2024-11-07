**Note**L The url will vary between LMStudio, Ollama, and Llamafile. 

* **lmstudio** : `http://127.0.0.1:1234/v1`
* **ollama**: `http://127.0.0.1:11434`
* **Llamafile**: `http://127.0.0.1:8080/v1`

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1", #assuming LLamafile
    api_key = "NoNeed"
)

# Function to generate an email
def generate_email(subject, recipient_name, additional_info):
    # Step 1: Create the prompt to guide the AI in generating the email
    prompt = f"Write a professional email to {recipient_name} with the subject '{subject}'. Include the following information: {additional_info}"
    
    # Step 2: Call the OpenAI API to generate the email
    # Hint: Use the 'chat' endpoint to create a completion based on the prompt
    response = client.chat.completions.create(
        model="TinyLLM",  # Replace "TinyLLama" with the correct model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=  300,# add a value,
        temperature= 0.4,# add a value,
        top_p= 0.4,# add a value,
        frequency_penalty= 0,# add a value,
        presence_penalty= 0,# add a value
    )
    
    # Step 3: Extract and return the generated email content from the API response
    return response.choices[0].message.content  # <-- Complete this line to get the email content

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

