# **Lab Instructions: Using LlamaIndex to Query a CSV Dataset**

## Step 1: Import Required Libraries

The code starts by installing the necessary packages:

```python
!pip install llama-index
!pip install llama-index-llms-llamafile
!pip install llama-index-embeddings-llamafile
```

- **llama-index**: This is the core package for LlamaIndex, which provides the framework for building and querying indexes.
- **llama-index-llms-llamafile**: This package integrates LLaMA models with LlamaIndex, allowing for the use of LLaMA models as the language model (LLM) component.
- **llama-index-embeddings-llamafile**: This package provides the embedding functionality using LLaMA models, which is essential for creating vector representations of documents.

Begin by importing the necessary libraries and modules from LlamaIndex:

```python
from llama_index.core import Settings
from llama_index.embeddings.llamafile import LlamafileEmbedding
from llama_index.llms.llamafile import Llamafile
from llama_index.readers.file import CSVReader
from llama_index.core import VectorStoreIndex
from pathlib import Path
```

## Step 2: Set Up LlamaIndex with LLaMAFile

Configure LlamaIndex to use the LLaMAFile for embeddings and language model functionalities:

```python
Settings.embed_model = LlamafileEmbedding(base_url="http://localhost:8080")
Settings.llm = Llamafile(base_url="http://localhost:8080", temperature=0, seed=0)
```

- **LlamafileEmbedding**: Initializes the embedding model using the LLaMAFile server.
- **Llamafile**: Sets up the language model with specified parameters like `temperature`and `seed`.

## Step 3: Load the CSV File

Use `CSVReader` to load data from the CSV file:

```python
file = Path('csvdata/shows.csv')
csv_reader = CSVReader()
docs = csv_reader.load_data(file=file)
```

- **CSVReader**: Reads data from the specified CSV file path.
- **load_data**: Loads the CSV content into a format suitable for indexing.

## Step 4: Build the Index

Create an index from the loaded documents:

```python
index = VectorStoreIndex.from_documents(docs)
```

- **VectorStoreIndex**: Constructs an index that stores vector representations of documents, enabling efficient querying.

## Step 5: Create a Query Engine

Transform the index into a query engine:

```python
query_engine = index.as_query_engine()
```

- **as_query_engine**: Converts the index into a query engine that can process queries and return relevant information.

## Step 6: Perform Queries

Define and execute queries to extract information from the dataset:

```python
# Query for main topic
query = "What is the main topic of the document?"
response = query_engine.query(query)
print(response)

# Query for most watched show
query = "What is the most watched show?"
response = query_engine.query(query)
print(response)
```

- **query**: A string representing the question or information you want to retrieve.
- **query_engine.query**: Executes the query against the index and returns a synthesized response.

## Expected Output

Running these queries should yield responses that summarize key aspects of your dataset, such as:

1. The main topic of the document, including details about shows and their rankings.
2. Identification of the most-watched show based on watchtime data.

By following these steps, you can effectively leverage LlamaIndex and LLaMAFile to process and analyze data stored in CSV format.