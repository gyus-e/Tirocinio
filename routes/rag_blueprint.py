from flask import Blueprint, Response, request, jsonify
from rag.RAGAgentManager import RAGAgentManager
import config


ragBlueprint = Blueprint("rag", __name__)


@ragBlueprint.post("/rag-settings")
def update_rag_config():
    data = request.get_json() if request.is_json else request.form

    if not data:
        return jsonify(success=False, message="Bad request"), 400

    config.EMBED_MODEL_NAME = int(data.get("embed_model_name", config.EMBED_MODEL_NAME))
    config.CONTEXT_WINDOW = int(data.get("context_window", config.CONTEXT_WINDOW))
    config.TEMPERATURE = float(data.get("temperature", config.TEMPERATURE))
    config.TOP_K = int(data.get("top_k", config.TOP_K))
    config.TOP_P = float(data.get("top_p", config.TOP_P))

    return jsonify(success=True), 200


@ragBlueprint.post("/rag-chat")
async def rag_chat():
    data = request.get_json() if request.is_json else request.form
    query = data.get("message", "").strip() if data else None

    if not data or not query:
        return jsonify(success=False, message="Bad request"), 400

    agent = await RAGAgentManager.get_rag_agent()
    context = await RAGAgentManager.get_context()
    answer = await agent.run(query, ctx=context)

    return jsonify(success=True, answer=answer), 200
