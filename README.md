# Prerequisites
- [Python](https://www.python.org/)
- pip

# Instructions
## (Optional) Create and activate venv
- `python -m venv .venv`

- On Windows: `.\.venv\Scripts\activate`
- On Linux/macOS: `source .venv/bin/activate`

## Install dependencies
- `pip install -r requirements-lock.txt`

Some libraries tend to change quite often, so stick to `requirements-lock.txt` to get a working version.
Only use `requirements.txt` if you know what you're doing and are prepared to change the code in case something breaks.

## Set HuggingFace token
Make sure you have a working [HuggingFace token](https://huggingface.co/).
Rename `.env.example` to `.env` and change the value of HF_TOKEN at the following line:
- ```HF_TOKEN=YOUR_HF_TOKEN```

You can optionally set other environment variables:
- `HF_HOME`: set the download path of HuggingFace models (defaults at: `C:\Users\<YourUsername>\.cache\huggingface\hub` on Windows; `~/.cache/huggingface/hub` on Linux/macOS);
- `OPENAI_API_KEY`: if you plan to use [OpenAI](https://openai.com/api/) models for RAG.

- Note: to make the software portable, it's recommended to set `HF_HOME` to a directory within the root of this application, such as `./models` (use absolute path).

## (Optional) Edit the configuration
You can change the values of the variables in the file `config.py` to your liking.

## Add the context documents
Delete the placeholder documents in the `documents` folder and replace them with the documents that will provide the context for the LLM's responses.
Supported formats are: `.txt|.json|.pdf`

## Run the script
- `python main.py`

## Cleanup
- You can clean the cache and the vector store by deleting the `storage` folder.
- You can safely delete all `__pycache__` folders.
- You can perform a clean install by deleting the `.venv` folder and repeating the instructions from the start.
- You can also delete the downloaded HuggingFace models (stored at the path pointed by the `HF_HOME` environment variable).