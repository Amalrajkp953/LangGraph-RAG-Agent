from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )

    return llm