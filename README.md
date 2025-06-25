# Prerequisites
[Python](https://www.python.org/)
pip

# Instructions

## (Optional) Create and activate venv
`python -m venv .venv`

On Linux:
`source .venv/bin/activate`

On Windows:
`.\venv\Scripts\activate`

## Install dependencies
`pip install -r requirements.txt`

## Set HuggingFace token
Make sure you have a working [HuggingFace token](https://huggingface.co/).
In the root directory, create a file named `hf_token.py` with the following line:
```HF_TOKEN = "YOUR_HF_TOKEN"```

## (Optional) Edit the default configuration
You can change the values of the variables in the file `params.py` to your liking.

## Run the script
`python main.py`