from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

model = ChatMistralAI(model="mistral-small-2506",temperature=0.7)

print("Choose your AI mode :\n1. Funny AI\n2. Sad AI\n3. Angry AI")

choice= int(input("Enter your choice (1/2/3): "))

if choice == 1:
   mode="You are a funny AI agent. You respond to all prompts with humor and wit."
elif choice == 2:
   mode="You are a sad AI agent. You respond to all prompts with empathy and understanding."
elif choice == 3:
   mode="You are an angry AI agent. You respond to all prompts with frustration and irritation."

messages = [
    SystemMessage(content=mode)
]

print("--------------------Welcome------------type 0 to exit--------------------")

while True:

    prompt = input("You : ")
    messages.append(HumanMessage(content=prompt))
    if prompt == "0":
       break
    response= model.invoke(messages)
    messages.append(AIMessage(content=response.content))

    print("Bot :",response.content)

print(messages)
