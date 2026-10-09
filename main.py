from dotenv import load_dotenv

load_dotenv()

from rag.loader import load_pdf
from rag.embeddings import get_embeddings
from rag.vector_store import create_vector_store
from rag.llm import get_llm
from rag.graph import create_graph


# --------------------------------------------------
# 1. Load PDF
# --------------------------------------------------

pages = load_pdf()

print("PDF loaded successfully!")
print(f"Number of pages: {len(pages)}")


# --------------------------------------------------
# 2. Create embeddings
# --------------------------------------------------

embeddings = get_embeddings()


# --------------------------------------------------
# 3. Create Gemini LLM
# --------------------------------------------------

llm = get_llm()


# --------------------------------------------------
# 4. Create vector store
# --------------------------------------------------

vector_store = create_vector_store(
    pages,
    embeddings
)

print("Vector store created successfully!")


# --------------------------------------------------
# 5. Create LangGraph
# --------------------------------------------------

graph = create_graph(
    vector_store,
    llm
)


# --------------------------------------------------
# 6. Conversation history
# --------------------------------------------------

history = []


# --------------------------------------------------
# 7. Chat loop
# --------------------------------------------------

while True:

    question = input(
        "\nAsk a question about the PDF "
        "(type 'exit' to quit): "
    )

    # --------------------------------------------------
    # Exit the application
    # --------------------------------------------------

    if question.lower() == "exit":

        print("\nExiting...")
        break


    # --------------------------------------------------
    # Ignore empty questions
    # --------------------------------------------------

    if not question.strip():

        print("\nPlease enter a question.")
        continue


    # --------------------------------------------------
    # Send question to LangGraph
    # --------------------------------------------------

    result = graph.invoke({

        "question": question,

        # Used by the query-rewriting node
        "search_query": "",

        # Retrieved PDF context
        "context": "",

        # Generated answer
        "answer": "",

        # Previous conversation
        "history": history,

        # Retrieved source information
        "sources": [],

        # Relevance result
        "relevance": "",

        # Internal retrieval candidates
        "_candidates": []
    })


    # --------------------------------------------------
    # Update conversation history
    # --------------------------------------------------

    history = result.get(
        "history",
        history
    )


    # --------------------------------------------------
    # Display answer
    # --------------------------------------------------

    print("\nAnswer:")
    print(result["answer"])


    # --------------------------------------------------
    # Display sources
    # --------------------------------------------------

    if result.get("sources"):

        print("\nSources:")

        for source in result["sources"]:

            print(
                f"- Page {source['page']} "
                f"(relevance: {source['score']:.0f})"
            )