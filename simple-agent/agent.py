import os
from dotenv import load_dotenv
# from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

load_dotenv()

# Initialize project client
project_client = AIProjectClient(
    endpoint=os.getenv("PROJECT_ENDPOINT"),
    # credential=AzureKeyCredential(os.getenv("API_KEY"))
    credential=DefaultAzureCredential()
)

# Use the OpenAI client in stable SDK
openai_client = project_client.get_openai_client()

model = os.getenv("MODEL_DEPLOYMENT_NAME")

print("Endpoint:", os.getenv("PROJECT_ENDPOINT"))
print("API Key:", os.getenv("API_KEY")[:5], "...")  # just show first few chars
print("Model:", os.getenv("MODEL_DEPLOYMENT_NAME"))

choice = ""

while choice != "END":
    user_query = input("Enter your query: ")

    # Send request to the model
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_query}
        ]
    )

    # Print the assistant’s reply
    print(response.choices[0].message.content)

    choice = input("Enter END to stop: ")

print("Conversation Ended")
