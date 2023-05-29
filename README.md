# EAGE Annual 2023 Hackathon Starter Pack

## Introduction
This repository contains a starter pack code for the EAGE annual 2023 hackathon.
There are 3 examples and two provided datasets for the EAGE Annual 2023 abstract library.

- VectorDB of embedded abstract text chunks: [create_abstract_chunk_embeddings.py](https://github.com/LukasMosser/EAGE-Hackathon-2023-Starter-Pack/blob/master/create_abstract_chunk_embeddings.py)
- VectorDB of paper summaries: [create_full_abstract_summary_embeddings.py](https://github.com/LukasMosser/EAGE-Hackathon-2023-Starter-Pack/blob/master/create_full_abstract_summary_embeddings.py)
- Slack bot with [modal](https://modal.com/): [summary_bot.py](https://github.com/LukasMosser/EAGE-Hackathon-2023-Starter-Pack/blob/master/summary_bot.py)

## Datasets
The EAGE has made available all of the abstracts from the EAGE Annual 2023.
[GDrive](https://drive.google.com/drive/folders/1zw9-kiMypuyj09aWBXQKKtik3wIQsCRK?usp=sharing)

From this two starter vector databases have been created.

These are intended to serve beginners or those that dont have the necessary API or computational resources to create their own embeddings.
- VectorDB of chunked paper parts: [GDrive](https://drive.google.com/drive/folders/1zw9-kiMypuyj09aWBXQKKtik3wIQsCRK?usp=sharing)
- VectorDB of paper summaries generated with langchain: [GDrive](https://drive.google.com/drive/folders/1zw9-kiMypuyj09aWBXQKKtik3wIQsCRK?usp=sharing)

Both vector DBs were created with the above python scripts.

## Data License
The data provided here is subject to license terms from the EAGE.
Embeddings and vector store are subject to OpenAI license terms for created embeddings.
Text is subject to permission from the EAGE to be used.

## Code License
All code is CC-BY-4.0
