
from typing import TypedDict, List
import re
import os

from langgraph.graph import StateGraph, END


# ==========================================================
# GRAPH STATE
# ==========================================================

class GraphState(TypedDict, total=False):
    question: str
    resolved_question: str
    search_query: str
    context: str
    answer: str
    history: List[str]
    sources: List[dict]
    relevance: str
    _candidates: list


FALLBACK_ANSWER = (
    "The PDF does not provide enough information "
    "to answer this question."
)


# ==========================================================
# HELPER: EXTRACT TEXT FROM GEMINI RESPONSE
# ==========================================================

def get_response_text(response) -> str:
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(item.text))

        return "\n".join(parts).strip()

    return str(content).strip()


# ==========================================================
# HELPER: COMPANY DETECTION
# ==========================================================

COMPANY_NAMES = [
    "nvidia",
    "microsoft",
    "alphabet",
    "facebook",
    "amazon",
    "netflix",
    "palantir",
    "apple",
    "google",
    "tesla",
    "meta",
    "ionq",
    "arm",
]

COMPANY_ALIASES = {
    "google": "alphabet",
    "facebook": "meta",
}

COMPANY_PATTERNS = {
    "alphabet": r"\b(alphabet|google)\b",
    "meta": r"\b(meta|facebook)\b",
    "arm": r"\b(arm holdings|arm)\b",
}


def find_company(text: str):
    text = text.lower()

    for name in sorted(COMPANY_NAMES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", text):
            return COMPANY_ALIASES.get(name, name)

    return None


def company_pattern(company):
    if not company:
        return None

    return COMPANY_PATTERNS.get(
        company,
        rf"\b{re.escape(company)}\b",
    )


def mentions_company(text: str, company: str) -> bool:
    pattern = company_pattern(company)

    return bool(
        pattern and re.search(pattern, text, re.IGNORECASE)
    )


# ==========================================================
# HELPER: SOURCE METADATA
# ==========================================================

def get_page_number(document) -> int:
    """Convert zero-based PDF page metadata to a page number."""
    try:
        return int(document.metadata.get("page", 0)) + 1
    except (TypeError, ValueError):
        return 1


def get_source_name(metadata: dict) -> str:
    """Keep source filenames for future multi-PDF support."""
    source = str(metadata.get("source", "Unknown document"))
    return os.path.basename(source.replace("\\", "/"))


def unique_sources(candidates: list) -> list:
    """Return one citation per source file and page."""
    sources = []
    seen = set()

    for item in candidates:
        source = item.get("source", "Unknown document")
        page = item.get("page", 1)
        key = (source, page)

        if key in seen:
            continue

        seen.add(key)

        sources.append({
            "source": source,
            "page": page,
            "score": item.get("score", 0.0),
        })

    return sources


# ==========================================================
# HELPER: FOLLOW-UP QUESTIONS
# ==========================================================

FOLLOW_UP_PHRASES = [
    "what about",
    "how about",
    "what was its",
    "what is its",
    "what were its",
    "how did it",
    "how did they",
    "tell me more",
    "explain more",
    "and its",
    "and what about",
    "its performance",
    "its revenue",
    "its growth",
    "their performance",
    "what about its",
    "what about their",
    "what drove that",
    "why did it",
    "what about them",
    "what happened to it",
    "and what happened",
]


def is_follow_up_question(question: str) -> bool:
    question = question.lower().strip()

    return any(
        phrase in question
        for phrase in FOLLOW_UP_PHRASES
    )


def find_previous_company(history: list):
    """Find the latest company mentioned in a user's question."""
    for entry in reversed(history[-10:]):
        if not isinstance(entry, str):
            continue

        if entry.lower().startswith("user:"):
            company = find_company(entry[5:].strip())

            if company:
                return company

    return None


def get_relevant_company(question: str, history: list):
    company = find_company(question)

    if company:
        return company

    if is_follow_up_question(question):
        return find_previous_company(history)

    return None


# ==========================================================
# QUERY RESOLUTION AND REWRITING
# ==========================================================

def rewrite_query_node(state: GraphState, llm):
    question = state.get("question", "").strip()
    history = state.get("history", [])

    company = get_relevant_company(question, history)
    resolved_question = question

    if (
        not find_company(question)
        and is_follow_up_question(question)
        and company
    ):
        history_text = "\n".join(history[-10:])

        prompt = f"""
Rewrite the latest user question as a standalone question.

Intended company: {company}

Conversation history:
{history_text}

Latest user question:
{question}

Rules:
- Preserve the user's intent.
- Replace pronouns with the intended company.
- Preserve requested years and metrics.
- Do not answer the question.
- Do not add facts.

Return only the rewritten question.
"""

        try:
            rewritten = get_response_text(
                llm.invoke(prompt)
            ).strip().strip('"')

            if rewritten:
                if not mentions_company(rewritten, company):
                    rewritten = f"{company}: {rewritten}"

                resolved_question = rewritten

        except Exception as exc:
            print(f"Follow-up resolution warning: {exc}")
            resolved_question = f"{company}: {question}"

    print(f"Resolved question: {resolved_question}")

    resolved_company = (
        find_company(resolved_question) or company
    )

    if resolved_company:
        search_query = (
            f"{resolved_company} {resolved_question} "
            "2024 stock performance financial results"
        )
    else:
        prompt = f"""
Rewrite this question as a clear search query for a PDF.

Question:
{resolved_question}

Do not answer the question.
Return only the search query.
"""

        try:
            search_query = get_response_text(
                llm.invoke(prompt)
            ).strip().strip('"')
        except Exception as exc:
            print(f"Query rewrite warning: {exc}")
            search_query = resolved_question

        if not search_query:
            search_query = resolved_question

    print(f"Search query: {search_query}")

    return {
        **state,
        "resolved_question": resolved_question,
        "search_query": search_query,
    }


# ==========================================================
# RETRIEVE DOCUMENTS
# ==========================================================

def retrieve_node(state: GraphState, vector_store):
    question = state.get(
        "resolved_question",
        state.get("question", ""),
    ).strip()

    search_query = state.get("search_query", question).strip()
    company = find_company(question)

    queries = [question, search_query]

    aliases = {
        "alphabet": ["Alphabet", "Google"],
        "meta": ["Meta", "Facebook"],
        "arm": ["Arm", "Arm Holdings"],
    }

    if company:
        for name in aliases.get(company, [company]):
            queries.extend([
                f"{name} 2024",
                f"{name} stock performance in 2024",
                f"{name} shares financial results 2024",
                f"{name} annual return share price earnings",
            ])

        queries.append(f"Magnificent Seven {company} 2024")

    candidates = []
    seen_chunks = set()

    for query in dict.fromkeys(queries):
        try:
            documents = vector_store.similarity_search(
                query,
                k=8,
            )
        except Exception as exc:
            print(f"Search warning for '{query}': {exc}")
            continue

        for document in documents:
            content = (document.page_content or "").strip()

            if not content:
                continue

            source = get_source_name(document.metadata)

            normalized = re.sub(
                r"\s+", " ", content
            ).lower()

            # Deduplicate identical text within the same file.
            chunk_key = (source, normalized)

            if chunk_key in seen_chunks:
                continue

            seen_chunks.add(chunk_key)

            candidates.append({
                "source": source,
                "page": get_page_number(document),
                "content": content,
                "score": 0.0,
            })

    print(
        f"Retrieved {len(candidates)} unique candidate chunks."
    )

    if company:
        matching_count = sum(
            1 for item in candidates
            if mentions_company(item["content"], company)
        )
        print(f"Chunks mentioning {company}: {matching_count}")

    return {
        **state,
        "_candidates": candidates,
        "sources": [],
    }


# ==========================================================
# RERANK DOCUMENTS
# ==========================================================

STOP_WORDS = {
    "the", "and", "for", "was", "what", "about", "with",
    "from", "this", "that", "its", "their", "how", "did",
    "does", "are", "is", "in", "on", "of", "to", "a",
    "an", "it", "they", "tell", "me", "more", "please",
    "explain", "give", "show", "you", "were", "who",
    "when", "where", "which", "can", "could", "would",
    "should", "do", "has", "have", "had",
}

FINANCIAL_TERMS = {
    "stock", "shares", "performance", "revenue",
    "earnings", "growth", "market", "return", "price",
    "profit", "income", "valuation", "sales", "margin",
    "financial", "investment",
}


def rerank_node(state: GraphState):
    question = state.get(
        "resolved_question",
        state.get("question", ""),
    )

    search_query = state.get("search_query", "")
    candidates = list(state.get("_candidates", []))
    company = find_company(question)

    query_text = f"{question} {search_query}"

    question_terms = {
        word.lower()
        for word in re.findall(r"[a-zA-Z0-9]+", query_text)
        if len(word) > 2 and word.lower() not in STOP_WORDS
    }

    scored = []

    for original in candidates:
        item = dict(original)
        content = item.get("content", "")
        content_lower = content.lower()

        matched_terms = {
            term for term in question_terms
            if re.search(
                rf"\b{re.escape(term)}\b",
                content_lower,
            )
        }

        score = len(matched_terms)

        company_match = (
            mentions_company(content, company)
            if company else False
        )

        if company_match:
            score += 8

        score += sum(
            1 for term in FINANCIAL_TERMS
            if term in question_terms
            and re.search(
                rf"\b{re.escape(term)}\b",
                content_lower,
            )
        )

        item["score"] = float(score)
        item["_company_match"] = company_match
        item["_matched_terms"] = len(matched_terms)

        if company:
            # Do not use unrelated companies as evidence.
            if company_match:
                scored.append(item)
        elif matched_terms:
            scored.append(item)

    scored.sort(
        key=lambda item: (
            item["score"],
            item["_matched_terms"],
        ),
        reverse=True,
    )

    # Avoid selecting several chunks from the same page
    # when other relevant pages are available.
    selected = []
    seen_pages = set()

    for item in scored:
        page_key = (item["source"], item["page"])

        if page_key in seen_pages:
            continue

        selected.append(item)
        seen_pages.add(page_key)

        if len(selected) == 4:
            break

    sources = unique_sources(selected)

    print("\nSelected sources:")

    for item in selected:
        print(
            f"- {item['source']} | Page {item['page']} "
            f"(score: {item['score']:.1f})"
        )

    if not selected:
        print("No sufficiently matching chunks found.")

    return {
        **state,
        "_candidates": selected,
        "sources": sources,
    }


# ==========================================================
# BUILD CONTEXT
# ==========================================================

def build_context_node(state: GraphState):
    context_parts = []

    for candidate in state.get("_candidates", []):
        context_parts.append(
            f"SOURCE: {candidate['source']}\n"
            f"PAGE: {candidate['page']}\n"
            f"{candidate['content']}"
        )

    return {
        **state,
        "context": "\n\n".join(context_parts),
    }


# ==========================================================
# CHECK RELEVANCE
# ==========================================================

def relevance_node(state: GraphState, llm):
    question = state.get(
        "resolved_question",
        state.get("question", ""),
    )

    context = state.get("context", "").strip()
    candidates = state.get("_candidates", [])

    if not context or not candidates:
        print("\nRelevance check: NOT_RELEVANT")
        return {
            **state,
            "relevance": "NOT_RELEVANT",
            "sources": [],
        }

    company = find_company(question)

    if company and not any(
        mentions_company(item.get("content", ""), company)
        for item in candidates
    ):
        print("\nRelevance check: NOT_RELEVANT")
        return {
            **state,
            "relevance": "NOT_RELEVANT",
            "sources": [],
        }

    prompt = f"""
Determine whether these PDF excerpts contain evidence useful
for answering the user's actual question.

Question:
{question}

PDF excerpts:
{context}

Rules:
- Related subject matter alone is not sufficient.
- Individual company questions require evidence about that company.
- Group statistics are not individual company statistics.
- Partial evidence can be relevant for a partial answer.

Return exactly one word:
RELEVANT
or
NOT_RELEVANT
"""

    try:
        result = get_response_text(
            llm.invoke(prompt)
        ).strip().upper()
    except Exception as exc:
        print(f"Relevance check warning: {exc}")
        return {
            **state,
            "relevance": "NOT_RELEVANT",
            "sources": [],
        }

    first_word = result.split()[0] if result.split() else ""

    relevance = (
        "RELEVANT"
        if first_word == "RELEVANT"
        else "NOT_RELEVANT"
    )

    print(f"\nRelevance check: {relevance}")

    if relevance == "NOT_RELEVANT":
        return {
            **state,
            "relevance": relevance,
            "sources": [],
        }

    return {
        **state,
        "relevance": relevance,
    }


# ==========================================================
# ROUTE AFTER RELEVANCE CHECK
# ==========================================================

def route_after_relevance(state: GraphState):
    if state.get("relevance") == "RELEVANT":
        return "generate"

    return "fallback"


# ==========================================================
# GENERATE ANSWER
# ==========================================================

def generate_node(state: GraphState, llm):
    original_question = state.get("question", "")

    question = state.get(
        "resolved_question",
        original_question,
    )

    context = state.get("context", "")

    prompt = f"""
Answer the user's question using only the PDF excerpts.

Original user question:
{original_question}

Resolved standalone question:
{question}

PDF excerpts:
{context}

Instructions:
1. Answer directly and concisely.
2. Use only the supplied excerpts for factual claims.
3. Do not guess or use outside knowledge.
4. Preserve figures, dates and percentages accurately.
5. Never substitute group statistics for individual performance.
6. If the company is mentioned but the requested metric is absent,
   explain what is supported and what is missing.
7. Never switch to another company.
8. If evidence is insufficient, say:
   {FALLBACK_ANSWER}
9. Cite only source filenames and page numbers present in the excerpts.
10. Do not invent citations.

Return the answer only.
"""

    try:
        answer = get_response_text(llm.invoke(prompt))

        if not answer:
            answer = FALLBACK_ANSWER

    except Exception as exc:
        print(f"Answer generation warning: {exc}")
        answer = FALLBACK_ANSWER

    return {
        **state,
        "answer": answer,
    }


# ==========================================================
# FALLBACK ANSWER
# ==========================================================

def fallback_node(state: GraphState):
    return {
        **state,
        "answer": FALLBACK_ANSWER,
        "sources": [],
    }


# ==========================================================
# UPDATE CONVERSATION HISTORY
# ==========================================================

def update_history_node(state: GraphState):
    history = list(state.get("history", []))

    history.append(f"User: {state.get('question', '')}")
    history.append(f"Assistant: {state.get('answer', '')}")

    return {
        **state,
        "history": history[-10:],
    }


# ==========================================================
# CREATE LANGGRAPH WORKFLOW
# ==========================================================

def create_graph(vector_store, llm):
    graph = StateGraph(GraphState)

    graph.add_node(
        "rewrite_query",
        lambda state: rewrite_query_node(state, llm),
    )

    graph.add_node(
        "retrieve",
        lambda state: retrieve_node(state, vector_store),
    )

    graph.add_node("rerank", rerank_node)
    graph.add_node("build_context", build_context_node)

    graph.add_node(
        "relevance",
        lambda state: relevance_node(state, llm),
    )

    graph.add_node(
        "generate",
        lambda state: generate_node(state, llm),
    )

    graph.add_node("fallback", fallback_node)
    graph.add_node("update_history", update_history_node)

    graph.set_entry_point("rewrite_query")

    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "build_context")
    graph.add_edge("build_context", "relevance")

    graph.add_conditional_edges(
        "relevance",
        route_after_relevance,
        {
            "generate": "generate",
            "fallback": "fallback",
        },
    )

    graph.add_edge("generate", "update_history")
    graph.add_edge("fallback", "update_history")
    graph.add_edge("update_history", END)

    return graph.compile()