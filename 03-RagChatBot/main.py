from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()





embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore=Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)
llm=ChatGroq(
     model="openai/gpt-oss-20b",
)

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",""" You are a spritual Guru

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."""),(
    "human","""Context:{context}
    
    Question:{question}"""
)
    ]
)

print("Meet you Guru! Ask me anything about spirituality and I will answer you based on the context provided in the documents. If the answer is not present in the documents, I will let you know that I could not find the answer in the document.")
print("Type 'exit' to quit. " )

while True:
    question=input("Ask your question: ")
    if question=="exit":
        print("GoodBye")
        break
    docs=retriever.invoke(question)
    context="\n\n".join([doc.page_content for doc in docs])
    final_prompt=prompt.invoke({
        "context":context,
        "question":question
    }
    )
    response=llm.invoke(final_prompt)
    print(f"Guru:{response.content}")

