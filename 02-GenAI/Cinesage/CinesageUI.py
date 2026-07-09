import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()
from langchain_groq import ChatGroq

llm = ChatGroq(
    model='openai/gpt-oss-20b',
    temperature=0.7,
    max_tokens=512,
)


class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    cast: List[str]
    director: Optional[str]
    summary: str


parser = PydanticOutputParser(pydantic_object=Movie)

system_prompt = SystemMessagePromptTemplate.from_template(
    """
    Extract movie information from the paragrapgh provided
    {format_instructions}
    """
)
user_prompt = HumanMessagePromptTemplate.from_template(
    "{paragraph}"
)
prompt = ChatPromptTemplate.from_messages([
    system_prompt, user_prompt
])

st.set_page_config(page_title="Movie Info Extractor")
st.title("Movie Info Extractor")

paragraph = st.text_area("Enter Movie paragraph:")

if st.button("Extract"):
    if paragraph.strip():
        response = llm.invoke(
            prompt.format(
                paragraph=paragraph,
                format_instructions=parser.get_format_instructions()
            )
        )
        movie_data = parser.parse(response.content)
        format_instructions=parser.get_format_instructions()
        st.write(format_instructions)
        st.write(movie_data)
    else:
        st.warning("Please enter a movie paragraph.")