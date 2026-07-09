from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=512,
)

print("Choose your AI mode")
print("1. Angry")
print("2. Sad")
print("3. Funny")

choice = int(input("Tell your response: "))

if choice == 1:
    mode = "You are an angry AI agent. Respond aggressively and impatiently."
elif choice == 2:
    mode = "You are a very sad AI agent. Respond emotionally and depressingly."
elif choice == 3:
    mode = "You are a funny and witty AI agent. Respond humorously and sarcastically."
else:
    mode = "You are a polite and neutral AI assistant."

messages = [
    SystemMessage(content=mode)
]

print("\nType 0 to exit.\n")

while True:

    prompt = input("You: ")

    if prompt == "0":
        break

    messages.append(HumanMessage(content=prompt))

    response = llm.invoke(messages)

    messages.append(AIMessage(content=response.content))

    print("Bot:", response.content)