from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("documents").load_data()
knowledge = "\n".join([doc.text for doc in documents])
