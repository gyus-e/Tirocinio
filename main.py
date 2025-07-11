import asyncio
import torch
from flask import Flask
from dotenv import load_dotenv
from transformers.cache_utils import DynamicCache

from utils import get_models_list, get_embed_models_list
from routes import cagBlueprint, ragBlueprint, settingsBlueprint


def main():
    load_dotenv()
    torch.serialization.add_safe_globals([DynamicCache])
    # torch.set_grad_enabled(False)

    print(get_models_list())
    print(get_embed_models_list())

    app = Flask(__name__)

    app.register_blueprint(settingsBlueprint)
    app.register_blueprint(ragBlueprint)
    app.register_blueprint(cagBlueprint)
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
