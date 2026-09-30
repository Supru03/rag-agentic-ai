import re
from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from src import config

REFUSAL = "I cannot answer based on the provided document."


class AgentState(TypedDict):
    question: str
    context: List[str]
    scores: List[float]
    answer: str
    score: float


def build_rag_graph():
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
    vectorstore = PineconeVectorStore(
        index_name=config.INDEX_NAME, embedding=embeddings
    )
    llm = ChatGroq(model=config.LLM_MODEL, temperature=0)

    def retrieve(state: AgentState):
        question = state["question"]

        # Remove phrases that match the book's title page instead of its content
        query = re.sub(
            r"\b(according to|in|from|based on)\s+the\s+(e-?book|document|pdf|book)\b",
            "",
            question,
            flags=re.IGNORECASE,
        ).strip(" ?.") or question

        results = vectorstore.similarity_search_with_score(query, k=15)

        # Drop duplicates and short chunks (title page, table of contents)
        seen, unique = set(), []
        for doc, score in results:
            text = doc.page_content.strip()
            if text in seen or len(text) < 300:
                continue
            seen.add(text)
            unique.append((doc, score))

        unique = unique[: config.TOP_K]
        return {
            "context": [doc.page_content for doc, _ in unique],
            "scores": [float(s) for _, s in unique],
        }

    def generate(state: AgentState):
        top_score = max(state["scores"], default=0.0)

        # Guardrail: nothing relevant retrieved, so refuse without calling the LLM
        if not state["context"] or top_score < config.MIN_RELEVANCE:
            return {"answer": REFUSAL, "score": round(top_score, 3)}

        context_str = "\n\n---\n\n".join(state["context"])
        prompt = f"""You are a strict assistant. Answer the question using ONLY the context below.
If the context does not contain enough information, reply exactly:
"{REFUSAL}"
Do not use outside knowledge.

Context:
{context_str}

Question: {state['question']}"""

        answer = llm.invoke(prompt).content.strip()
        score = 0.0 if REFUSAL in answer else round(top_score, 3)
        return {"answer": answer, "score": score}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()