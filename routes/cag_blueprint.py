import torch
from flask import Blueprint, Response, request, jsonify
from environ import CACHE_PATH


cagBlueprint = Blueprint("cag", __name__)


@cagBlueprint.post("/cag-chat")
async def cag_chat():
    from cag.cag import ModelConfiguration, get_answer, clean_up_cache

    data = request.get_json() if request.is_json else request.form
    query = data.get("message", "").strip() if data else None

    if not data or not query:
        return jsonify(success=False, message="Bad request"), 400


    model_configuration = ModelConfiguration()
    model = model_configuration.model()
    tokenizer = model_configuration.tokenizer()
    torch_device = model_configuration.torch_device()

    loaded_cache = torch.load(CACHE_PATH)
    answer = get_answer(query, tokenizer, model, torch_device, loaded_cache)
    clean_up_cache(loaded_cache)


    return jsonify(success=True, answer=answer), 200