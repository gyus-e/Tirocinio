from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("documents").load_data()
