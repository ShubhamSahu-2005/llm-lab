import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, "chroma_db")


def prepare_chunks(documents):
    cleaned_docs = [doc for doc in documents if getattr(doc, "page_content", "").strip()]

    if not cleaned_docs:
        raise ValueError("The uploaded PDF does not contain any readable text.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(cleaned_docs)
    valid_chunks = [chunk for chunk in chunks if getattr(chunk, "page_content", "").strip()]

    if not valid_chunks:
        raise ValueError("The PDF text could not be split into usable chunks.")

    return valid_chunks


def build_vectorstore(file_path, persist_directory=PERSIST_DIRECTORY):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    chunks = prepare_chunks(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    vectorstore.persist()
    return vectorstore


def get_retriever(persist_directory=PERSIST_DIRECTORY):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )


def main():
    st.set_page_config(page_title="RAG Book Assistant")

    st.title("📚 RAG Book Assistant")
    st.write("Upload a PDF and ask questions from the document")

    uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        st.success("PDF uploaded successfully!")

        if st.button("Create Vector Database"):
            with st.spinner("Processing document..."):
                try:
                    build_vectorstore(file_path)
                except Exception as exc:
                    st.error(f"Could not build the vector database: {exc}")
                else:
                    st.success("Vector database created!")

    if os.path.exists(PERSIST_DIRECTORY):
        retriever = get_retriever()

        llm = ChatMistralAI(model="mistral-small-2506")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
                ),
                (
                    "human",
                    """Context:
{context}

Question:
{question}
""",
                ),
            ]
        )

        st.divider()
        st.subheader("Ask Questions From the Book")

        query = st.text_input("Enter your question")

        if query:
            docs = retriever.invoke(query)
            context = "\n\n".join(
                [doc.page_content for doc in docs if getattr(doc, "page_content", "").strip()]
            )

            if not context:
                st.warning("No relevant context was found for that question.")
                st.stop()

            final_prompt = prompt.invoke({
                "context": context,
                "question": query,
            })

            response = llm.invoke(final_prompt)

            st.write("### AI Answer")
            st.write(response.content)


if __name__ == "__main__":
    main()