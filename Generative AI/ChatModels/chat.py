from dotenv import load_dotenv
load_dotenv()

#from langchain.chat_models import init_chat_model

#model = init_chat_model("gpt-5")

from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

response= model.invoke("What is football?")

print(response.content)
#import os
#print(os.getenv("OPENAI_API_KEY")[:10])