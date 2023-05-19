import os
from typing import Optional
from fastapi import Request
from modal import Image, Secret, Mount, Stub, web_endpoint

stub = Stub("eage-annual-2023-hackathon-summary-bot")

CACHE_PATH = "/root/eage_annual_2023_summaries_basic_test"


@stub.function(
    image=(
        Image.debian_slim().pip_install(
            "openai",
            "langchain",
            "tiktoken",
            "pypdf",
            "faiss-cpu",
            "slack-sdk",
            "fastapi",
        )
    ),
    secret=Secret.from_name("OPENAI_API_KEY"),
    mounts=[
        Mount.from_local_dir(
            "data/vectordb/eage_annual_2023_summaries_basic_test",
            remote_path="/root/eage_annual_2023_summaries_basic_test",
        )
    ],
)
async def run_slack_bot(question: str, channel_name: Optional[str] = None):
    import openai
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS

    # Create a base embeddings api object
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("/root/eage_annual_2023_summaries_basic_test", embeddings)

    query_result = db.similarity_search(question, k=7)

    context = " ".join(
        [
            "Summary {0:}: {1:}".format(i, q.page_content)
            for i, q in enumerate(query_result)
        ]
    )

    completion = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"The following content is contextual information: {context} \n"
        f"Please answer the following question based on the context: {question} \n"
        f"If you can't answer based on the context say: "
        f"I am sorry but I am unable to answer.",
        max_tokens=500,
    )

    result = completion.choices[0].text

    if channel_name:
        write_query_to_slack.call(channel_name, result)

    return query_result


@stub.function()
@web_endpoint(method="POST")
async def entrypoint(request: Request):
    body = await request.form()
    prompt = body["text"]
    run_slack_bot.spawn(prompt, body["channel_name"])
    return f"Running query for: {prompt}."


@stub.function(
    image=Image.debian_slim().pip_install("slack-sdk"),
    secret=Secret.from_name("eage-summary-bot-token"),
)
def write_query_to_slack(channel_name: str, query_result: str):
    import slack_sdk

    client = slack_sdk.WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    client.chat_postMessage(channel=channel_name, text=query_result)


@stub.local_entrypoint()
def run(prompt: str = "What abstracts are related to CO2 sequestration"):
    query_result = run_slack_bot.call(prompt)

    print(f"{query_result}")
