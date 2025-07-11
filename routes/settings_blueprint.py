from flask import Blueprint, Response, request, jsonify
import config


settingsBlueprint = Blueprint("settings", __name__)


@settingsBlueprint.post("/settings")
def update_config():
    data = request.get_json() if request.is_json else request.form

    if not data or not data.get("system_prompt"):
        return jsonify(success=False, message="Bad request"), 400

    if not data.get("model_name"):
        print(
            f"Warning: no Model Name set. Initializing with default model: ${config.MODEL_NAME}."
        )

    config.SYSTEM_PROMPT = data.get("system_prompt", config.SYSTEM_PROMPT)
    config.MODEL_NAME = data.get("model_name", config.MODEL_NAME)

    print("system_prompt", config.SYSTEM_PROMPT, "model_name", config.MODEL_NAME)
    return jsonify(success=True), 200