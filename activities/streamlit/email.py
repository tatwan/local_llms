from openai import OpenAI
import streamlit as st

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1", #assuming LLamafile
    api_key = "NoNeed"
)

st.title("Welcome to the Email Writing Assistant!")
st.caption("This is a chatbot powered by TinyLLM running locally on your machine.")


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
        max_tokens=  500,# add a value,
        temperature= 0.4,# add a value,
        top_p= 0.4,# add a value,
        frequency_penalty= 0,# add a value,
        presence_penalty= 0,# add a value
    )
    
    # Step 3: Extract and return the generated email content from the API response
    return response.choices[0].message.content  # <-- Complete this line to get the email content

# Function to start the email writing assistant
def capture_email_info():
    subject = st.text_input("Enter the email subject: ")
    recipient_name = st.text_input("Enter the recipient's name: ")
    additional_info = st.text_input("Enter any additional information to include in the email: ")
    return subject, recipient_name, additional_info
    
    # Step 5: Call the 'generate_email' function and display the generated email


subject, recipient_name, additional_info = capture_email_info()
email = generate_email(subject, recipient_name, additional_info)
if st.button("Generate Email"):
    email = generate_email(subject, recipient_name, additional_info)
    st.write(f"Generated Email: {email}")

