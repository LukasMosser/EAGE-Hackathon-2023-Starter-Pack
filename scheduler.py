import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd

st.set_page_config(layout="wide")


@st.cache_resource
def get_db():
    # Create a base embeddings api object
    embeddings = OpenAIEmbeddings()

    # Load the vector db
    db = FAISS.load_local(
        "./data/vectordb/eage_annual_2023_summaries_basic_test", embeddings
    )
    return db


@st.cache_resource
def get_metadata():
    df = pd.read_csv("./data/Annual 2023_Hackathon metadata.xlsx - Export.csv")
    return df


df = get_metadata()
db = get_db()


def ask_question(question: str, num_sessions: int):
    # query the vector db
    query_result = db.similarity_search(question, k=num_sessions)

    data = []
    for doc in query_result:
        fname = doc.metadata["source"].split("\\")[2]
        row = df.loc[df["File name"] == fname]
        data.append(
            {
                "summary": doc.page_content,
                "pdf": fname,
                "session_name": row.iloc[0]["Session Name"],
                "Paper Reference": row.iloc[0]["Paper Reference"],
                "title": row.iloc[0]["Title"],
                "authors": row.iloc[0]["Authors"],
                "eage_summary": row.iloc[0]["Summary"],
            }
        )
    return data


def make_markdown_template(
    title: str = None,
    summary: str = None,
    session_name: str = None,
    authors: str = None,
    eage_summary: str = None,
    pdf: str = None,
    **kwargs,
):
    template = (
        f"# Session Name:  \n"
        f"{session_name}  \n"
        f"# Title:  \n"
        f"{title}  \n"
        f"## Authors:  \n"
        f"{authors}   \n"
        f"## Summary  \n"
        f"{summary}  \n"
        f"## EAGE Summary  \n"
        f"{eage_summary}  \n"
    )
    return template


st.title("EAGE Annual 2023 - Agenda Builder")
question = st.text_input(
    label="What question are you here at EAGE to answer for you or your organization?"
)
num_sessions = st.slider(
    label="How many presentations would you like to be proposed with?",
    min_value=1,
    max_value=25,
    value=10,
)

if question:
    data = ask_question(question, num_sessions=num_sessions)

    for d in data:
        with st.expander(f"{d['session_name']} - {d['title']}"):
            st.markdown(make_markdown_template(**d))
