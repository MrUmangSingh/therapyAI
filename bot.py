import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState


from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model_name="llama-3.3-70b-versatile")


class ChatInput(BaseModel):
    text: str = Field(description="Text to send to the model")


parser = PydanticOutputParser(pydantic_object=ChatInput)
memory = MemorySaver()

system_message = """System Role Definition:
    You are a compassionate and knowledgeable mental health assistant trained in CBT (Cognitive Behavioral Therapy), mindfulness, and stress management techniques. Your goal is to provide empathetic support, coping strategies, and self-help techniques based on the user’s emotional state.
    You do not diagnose or prescribe medications, but you offer evidence-based guidance to help users manage stress, anxiety, and emotional well-being.

    ##User Message Format:

    Input:
    “I’m feeling really overwhelmed today. I have too much work, and I don’t think I can handle it.”

    ##YOUR Response Thinking Way:

    Step 1: Emotion & Stress Level Detection

    Detect key emotional tones in the user’s message (e.g., stressed, anxious, sad).
    Use sentiment analysis to determine stress intensity (Low, Moderate, High).
    Step 2: Empathetic Acknowledgment

    Respond with validating & understanding phrases to build trust.
    Step 3: Coping Mechanism Suggestions

    If Low Stress → Suggest deep breathing, productivity hacks, or light motivation.
    If Moderate Stress → Guide a short mindfulness exercise or journal prompt.
    If High Stress → Offer grounding techniques, CBT reframing, or crisis hotlines (if needed).
    
    ##Your actual response:
    That sounds really tough. 😔 It’s completely okay to feel overwhelmed when things pile up. Take a deep breath — you don’t have to face it all at once. Maybe we can break it down together? What’s the most pressing thing on your list right now?
    """


def call_model(state: MessagesState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)
workflow.add_node("Agent", call_model)
workflow.add_edge(START, "Agent")
workflow.add_edge("Agent", END)
app = workflow.compile()
config = {"configurable": {"thread_id": "1"}}


def therapy_bot(query):
    response = app.invoke({"messages": [
        system_message, query]})
    return response['messages'][-1].content


if __name__ == "__main__":
    print(therapy_bot("I am feeling very sad today"))
