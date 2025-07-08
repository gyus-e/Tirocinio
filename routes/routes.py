from flask import Blueprint, Response, request, jsonify
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow

import config
from rag.RAGAgentManager import RAGAgentManager

settingsBlueprint = Blueprint("settings", __name__)
ragBlueprint = Blueprint("rag", __name__)
cagBlueprint = Blueprint("cag", __name__)


@settingsBlueprint.post("/settings")
def update_config():
    data = request.get_json() if request.is_json else None

    if not data or not data.get("system_prompt"):
        return jsonify(success=False, message="Bad request"), 400

    if not data.get("model_name"):
        print(
            f"Warning: no Model Name set. Initializing with default model: ${config.MODEL_NAME}."
        )

    config.SYSTEM_PROMPT = data.get("system_prompt", config.SYSTEM_PROMPT)
    config.MODEL_NAME = data.get("model_name", config.MODEL_NAME)

    return jsonify(success=True), 200


@ragBlueprint.post("/rag-settings")
def update_rag_config():
    data = request.get_json() if request.is_json else None

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
    data = request.get_json() if request.is_json else None
    query = data.get("message", "").strip() if data else None

    if not data or not query:
        return jsonify(success=False, message="Bad request"), 400

    agent = await RAGAgentManager.get_rag_agent()
    context = await RAGAgentManager.get_context()
    answer = await agent.run(query, ctx=context)

    return jsonify(success=True, answer=answer), 200
