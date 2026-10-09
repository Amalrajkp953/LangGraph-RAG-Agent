def retrieve_documents(vector_store, question):

    print("\n--- RETRIEVAL DEBUG ---")
    print("Search query:", question)

    results = vector_store.similarity_search(
        question,
        k=8
    )

    for i, result in enumerate(results, 1):

        page = result.metadata.get("page", "Unknown")

        if isinstance(page, int):
            page += 1

        print(f"\nResult {i}")
        print("Page:", page)
        print("Content:")
        print(result.page_content[:500])

    print("\n--- END DEBUG ---")

    return results