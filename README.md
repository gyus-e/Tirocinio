# Requirements
 - Python 3.13
 - pip 25.3
 - Docker

# Warning
Running this software will download the LLM specified in "model.py" on your machine from HuggingFace.
It can be found in `~/.cache/huggingface/hub`

# Instructions
- Register to HuggingFace and get your HF_TOKEN.
- Rename `.env.example` to `.env` and change `YOUR_HF_TOKEN` with your actual HF_TOKEN.
- (Optional) create and activate a virtual environment:
    ```
        python -m venv .venv
        source .venv/bin/activate   # For Linux/MacOS users
        .\.venv\Scripts\activate    # For Windows users
    ```
- Install the requirements: 
    ```
        pip install -r requirements-lock.txt
    ```
- (Optional) Start the chroma container if you want persistent storage for your embeddings
    ```
        docker run --name chroma -p 8000:8000 -v ./chroma/data:/data -v ./chroma/config.yaml:/config.yaml ghcr.io/chroma-core/chroma:1.3.5
    ```
- (Optional) Replace the content of the `documents` folder with your desired context
- (Optional) Tweak the settings in `config.py`
- Run
    ```
        python main.py
    ```