from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import RunnableParallel

# -------------------------------------------------
# Load Environment Variables
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# Initialize LLM
# -------------------------------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

# -------------------------------------------------
# Pydantic Model
# -------------------------------------------------
class Book(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    author: List[str]
    summary: str
    price: Optional[float] = None

# -------------------------------------------------
# Create Structured LLM
# -------------------------------------------------
structured_llm = llm.with_structured_output(Book)

# -------------------------------------------------
# Prompts
# -------------------------------------------------
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert book information extractor."
)

summary_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    HumanMessagePromptTemplate.from_template(
        """
Book Description:

{text}

Write a summary under 100 words.
"""
    ),
])

details_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    HumanMessagePromptTemplate.from_template(
        """
Extract all information from the following text.

{text}

If any field is missing, return null.
"""
    ),
])

# -------------------------------------------------
# Chains
# -------------------------------------------------
short_chain = summary_prompt | llm
detail_chain = details_prompt | structured_llm

parallel_chain = RunnableParallel({
    "summary": short_chain,
    "details": detail_chain,
})

# -------------------------------------------------
# Input
# -------------------------------------------------
input_text = """
The book 'The Great Gatsby' by F. Scott Fitzgerald,
published in 1925, is a classic novel set in the Jazz Age.
It tells the story of Jay Gatsby's unrequited love for Daisy Buchanan
and explores themes of wealth, society, and the American Dream.
"""

# -------------------------------------------------
# Invoke
# -------------------------------------------------
response = parallel_chain.invoke({
    "text": input_text
})

# -------------------------------------------------
# Output
# -------------------------------------------------
print("=" * 60)
print("SHORT SUMMARY")
print("=" * 60)
print(response["summary"].content)

print("\n" + "=" * 60)
print("BOOK DETAILS")
print("=" * 60)

book = response["details"]

print(f"Title         : {book.title}")
print(f"Author        : {book.author}")
print(f"Release Year  : {book.release_year}")
print(f"Genre         : {book.genre}")
print(f"Summary       : {book.summary}")
print(f"Price         : {book.price}")