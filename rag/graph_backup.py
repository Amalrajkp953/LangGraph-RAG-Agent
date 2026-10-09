from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List


# ==========================================================
# GRAPH STATE
# ==========================================================

class GraphState(TypedDict):
    question: str
    search_query: str
    context: str
    answer: str
    history: List[str]
    sources: List[str]
    relevance: str
    _candidates: list


# ==========================================================
# 1. REWRITE USER QUERY
# ==========================================================

def rewrite_query_node(state, llm):

    question = state["question"]
    history = state.get("history", [])

    if not history:
        return {
            "search_query": question
        }

    previous_conversation = "\n".join(history)

    prompt = f"""
You are a query rewriting assistant for a PDF
question-answering system.

Convert the user's latest question into a standalone
search query that can be used to search the PDF.

Use the previous conversation to understand references
such as:

"What about Nvidia?"
"How did it perform?"
"What was its growth?"
"What about the company?"

If the question is already standalone,
keep its meaning unchanged.

Return ONLY the search query.
Do not explain anything.

Previous conversation:
{previous_conversation}

Latest user question:
{question}
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, list):

        search_query = "".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict)
        )

    else:
        search_query = response.content

    search_query = search_query.strip()

    print("\n========== QUERY REWRITE ==========")
    print("Original:", question)
    print("Search query:", search_query)
    print("===================================\n")

    return {
        "search_query": search_query
    }


# ==========================================================
# 2. RETRIEVE DOCUMENTS FROM CHROMADB
# ==========================================================

def retrieve_node(state, vector_store):

    search_query = state.get(
        "search_query",
        state["question"]
    )

    print("\n========== RETRIEVAL ==========")
    print("Search query:", search_query)

    results = vector_store.similarity_search(
        search_query,
        k=8
    )

    unique_results = []
    seen_chunks = set()

    for result in results:

        content = result.page_content.strip()

        if content in seen_chunks:
            continue

        seen_chunks.add(content)

        unique_results.append(
            result
        )

    print(
        "Unique retrieved chunks:",
        len(unique_results)
    )

    candidates = []

    for i, result in enumerate(
        unique_results,
        1
    ):

        page = result.metadata.get(
            "page",
            "Unknown"
        )

        if isinstance(page, int):
            page_number = page + 1
        else:
            page_number = page

        content = result.page_content.strip()

        candidates.append({
            "index": i,
            "page": page_number,
            "content": content
        })

        print(
            f"\nCandidate {i} "
            f"(Page {page_number})"
        )

        print(
            content[:300]
        )

    print("\n==============================\n")

    return {
        "context": "",
        "sources": [],
        "_candidates": candidates
    }


# ==========================================================
# 3. STRICT RERANK RETRIEVED DOCUMENTS
# ==========================================================

def rerank_node(state, llm):

    candidates = state.get(
        "_candidates",
        []
    )

    search_query = state.get(
        "search_query",
        state["question"]
    )

    if not candidates:

        return {
            "context": "",
            "sources": []
        }

    candidate_text = ""

    for candidate in candidates:

        candidate_text += (
            f"\n\n"
            f"CHUNK {candidate['index']}\n"
            f"PAGE {candidate['page']}\n"
            f"{candidate['content']}"
        )

    prompt = f"""
You are a strict document retrieval reranker.

Your job is to select ONLY the document chunks that
are directly useful for answering the user's question.

Search query:
{search_query}

Candidate document chunks:
{candidate_text}

IMPORTANT RULES:

1. Select the BEST 3 chunks at most.

2. Prefer chunks that directly discuss the exact
   company, stock, person, topic, number, or subject
   mentioned in the question.

3. If the question is about a specific company,
   strongly prefer chunks that explicitly mention
   that company.

4. DO NOT select a chunk about another company just
   because it is from the same industry, sector,
   page, or general topic.

5. General context may be selected only when it
   clearly helps answer the question.

6. If the question is about Nvidia, a chunk about
   Arm, Tesla, Meta, Apple, or another company should
   NOT be selected unless it contains directly useful
   information about Nvidia.

7. Do not select irrelevant chunks simply to reach
   three results.

8. If only one or two chunks are genuinely relevant,
   return only those chunks.

9. Do not invent information.

10. The chunk must contain information that can
    actually help answer the user's question.

Return ONLY the chunk numbers separated by commas.

Example:
4, 5

Do not explain your answer.
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, list):

        ranking_text = "".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict)
        )

    else:
        ranking_text = response.content

    ranking_text = ranking_text.strip()

    print("\n========== RERANKING ==========")
    print("Gemini ranking:", ranking_text)

    selected_numbers = []

    for part in ranking_text.replace(
        "\n",
        " "
    ).split(","):

        part = part.strip()

        digits = "".join(
            char
            for char in part
            if char.isdigit()
        )

        if digits:

            number = int(digits)

            if number not in selected_numbers:

                selected_numbers.append(
                    number
                )

    valid_numbers = {
        candidate["index"]
        for candidate in candidates
    }

    selected_numbers = [
        number
        for number in selected_numbers
        if number in valid_numbers
    ]

    # ------------------------------------------------------
    # FALLBACK
    # ------------------------------------------------------

    if not selected_numbers:

        selected_numbers = [
            candidate["index"]
            for candidate in candidates[:3]
        ]

    selected_numbers = selected_numbers[:3]

    selected_candidates = [
        candidate
        for candidate in candidates
        if candidate["index"]
        in selected_numbers
    ]

    context_parts = []
    sources = []

    for candidate in selected_candidates:

        context_parts.append(
            f"[Page {candidate['page']}]\n"
            f"{candidate['content']}"
        )

        sources.append(
            f"Page {candidate['page']}"
        )

    context = "\n\n".join(
        context_parts
    )

    sources = list(
        dict.fromkeys(sources)
    )

    print(
        "Selected chunks:",
        selected_numbers
    )

    print(
        "Selected sources:",
        sources
    )

    print("===============================\n")

    return {
        "context": context,
        "sources": sources
    }


# ==========================================================
# 4. CHECK RELEVANCE
# ==========================================================

def check_relevance_node(state, llm):

    search_query = state.get(
        "search_query",
        state["question"]
    )

    if not state.get("context"):

        return {
            "relevance": "NOT_RELEVANT"
        }

    prompt = f"""
You are a relevance checker for a PDF
question-answering system.

Determine whether the selected PDF context contains
useful information for answering the search query.

Return ONLY one of:

RELEVANT

or

NOT_RELEVANT

Return RELEVANT if the context contains information
that could reasonably help answer the query.

Search query:
{search_query}

Original question:
{state["question"]}

Selected PDF context:
{state["context"]}
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, list):

        text = "".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict)
        )

    else:
        text = response.content

    text = text.strip().upper()

    print("\n========== RELEVANCE ==========")
    print("Model response:", text)

    if "NOT_RELEVANT" in text:

        relevance = "NOT_RELEVANT"

    elif "RELEVANT" in text:

        relevance = "RELEVANT"

    else:

        relevance = "RELEVANT"

    print(
        "Final relevance:",
        relevance
    )

    print("===============================\n")

    return {
        "relevance": relevance
    }


# ==========================================================
# 5. ROUTE AFTER RELEVANCE CHECK
# ==========================================================

def route_after_relevance(state):

    if state["relevance"] == "RELEVANT":

        return "generate"

    return "fallback"


# ==========================================================
# 6. GENERATE FINAL ANSWER
# ==========================================================

def generate_node(state, llm):

    history = "\n".join(
        state.get("history", [])
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question ONLY using the information
provided in the selected PDF context.

Rules:

1. Do not use outside knowledge.

2. Do not invent information.

3. Answer directly and clearly.

4. If a number or percentage is present,
   include it accurately.

5. Use previous conversation only to understand
   references in follow-up questions.

6. If the context does not contain enough information,
   say:

"I couldn't find this information in the provided PDF."

Previous conversation:
{history}

Selected PDF context:
{state["context"]}

User question:
{state["question"]}
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, list):

        answer = "".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict)
        )

    else:

        answer = response.content

    answer = answer.strip()

    print("\n========== ANSWER ==========")
    print(answer)
    print("============================\n")

    new_history = (
        state.get("history", [])
        + [
            f"Q: {state['question']}\n"
            f"A: {answer}"
        ]
    )

    return {
        "answer": answer,
        "history": new_history
    }


# ==========================================================
# 7. FALLBACK
# ==========================================================

def fallback_node(state):

    answer = (
        "I couldn't find this information "
        "in the provided PDF."
    )

    print("\n========== FALLBACK ==========")
    print(answer)
    print("==============================\n")

    return {
        "answer": answer,
        "sources": [],
        "history": (
            state.get(
                "history",
                []
            )
            + [
                f"Q: {state['question']}\n"
                f"A: {answer}"
            ]
        )
    }


# ==========================================================
# 8. CREATE LANGGRAPH WORKFLOW
# ==========================================================

def create_graph(vector_store, llm):

    graph = StateGraph(
        GraphState
    )

    # ------------------------------------------------------
    # Nodes
    # ------------------------------------------------------

    graph.add_node(
        "rewrite_query",
        lambda state:
            rewrite_query_node(
                state,
                llm
            )
    )

    graph.add_node(
        "retrieve",
        lambda state:
            retrieve_node(
                state,
                vector_store
            )
    )

    graph.add_node(
        "rerank",
        lambda state:
            rerank_node(
                state,
                llm
            )
    )

    graph.add_node(
        "check_relevance",
        lambda state:
            check_relevance_node(
                state,
                llm
            )
    )

    graph.add_node(
        "generate",
        lambda state:
            generate_node(
                state,
                llm
            )
    )

    graph.add_node(
        "fallback",
        fallback_node
    )

    # ------------------------------------------------------
    # Edges
    # ------------------------------------------------------

    graph.add_edge(
        START,
        "rewrite_query"
    )

    graph.add_edge(
        "rewrite_query",
        "retrieve"
    )

    graph.add_edge(
        "retrieve",
        "rerank"
    )

    graph.add_edge(
        "rerank",
        "check_relevance"
    )

    # ------------------------------------------------------
    # Conditional Routing
    # ------------------------------------------------------

    graph.add_conditional_edges(
        "check_relevance",
        route_after_relevance,
        {
            "generate": "generate",
            "fallback": "fallback"
        }
    )

    # ------------------------------------------------------
    # End
    # ------------------------------------------------------

    graph.add_edge(
        "generate",
        END
    )

    graph.add_edge(
        "fallback",
        END
    )

    return graph.compile()