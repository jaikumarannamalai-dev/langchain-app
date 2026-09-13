from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Create the Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

# User input
user_input = "Explain LangChain in simple terms for a beginner."

# Send input to Gemini through LangChain
response = model.invoke(user_input)

# Display the result
print("=" * 60)
print("LangChain + Gemini + LangSmith Demo")
print("=" * 60)

print("\nInput:")
print(user_input)

print("\nModel Output:")
if isinstance(response.content, list):
    for item in response.content:
        if item.get("type") == "text":
            print(item.get("text"))
else:
    print(response.content)

print("\nDemo completed successfully.")