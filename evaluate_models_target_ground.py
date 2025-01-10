import os
from typing import Any, Dict, List, Optional
import json
import pandas as pd
from pathlib import Path

from azure.ai.evaluation import evaluate
from azure.ai.evaluation import GroundednessEvaluator
from azure.ai.evaluation.simulator import Simulator

import importlib.resources as pkg_resources

from app_target import ModelEndpoints

# %%
env_var = {
    "onnx-model": {
        "endpoint": "http://127.0.0.1:8000/score",
        "key": "",
    },
}

# %%
azure_ai_project = {
    "subscription_id": "3905431d-c062-4c17-8fd9-c51f89f334c4",
    "resource_group_name": "yangselenaai",
    "project_name": "azure_ai_studio_sdk",
}



# %%
import os

# Use the following code to set the environment variables if not already set. If set, you can skip this step.
os.environ["AZURE_OPENAI_API_KEY"] = "0152bce79cdf40adab70375917f4b8ec"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-06-01"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4o-mini"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ai-yangselenaai3739831789912690.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview"


# %%

model_config = {
    "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
    #"api_key": os.environ.get("AZURE_OPENAI_KEY"),
    "azure_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
}


# %%
resource_name = "grounding.json"
package = "azure.ai.evaluation.simulator._data_sources"
conversation_turns = []

with pkg_resources.path(package, resource_name) as grounding_file, Path.open(grounding_file, "r") as file:
    data = json.load(file)

for item in data:
    conversation_turns.append([item])
    if len(conversation_turns) == 2:
        break

# %%
async def custom_simulator_callback(
    messages: List[Dict],
    stream: bool = False,
    session_state: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> dict:
    messages_list = messages["messages"]
    # get last message
    latest_message = messages_list[-1]
    query = latest_message["content"]
    context = latest_message.get("context", None)
    # call your endpoint or ai application here
    model = "onnx-model"
    target=ModelEndpoints(env_var, model)
    response = target(query)["response"]
    # we are formatting the response to follow the openAI chat protocol format
    message = {
        "content": response,
        "role": "assistant",
        "context": context,
    }
    messages["messages"].append(message)
    return {"messages": messages["messages"], "stream": stream, "session_state": session_state, "context": context}

async def async_main_ground():
    # %%
    custom_simulator = Simulator(model_config=model_config)
    outputs = await custom_simulator(
        target=custom_simulator_callback,
        conversation_turns=conversation_turns,
        max_conversation_turns=1,
        concurrent_async_tasks=10,
    )

    print(outputs)


    # %%


    output_file = "outputs_ground.jsonl"
    with Path.open(output_file, "w") as file:
        for output in outputs:
            file.write(output.to_eval_qr_json_lines())

    # %%
    filepath = 'outputs_ground.jsonl'
    df = pd.read_json(filepath, lines=True)
    print(df.head())

    # %%
    groundedness_evaluator = GroundednessEvaluator(model_config=model_config)
    eval_output = evaluate(
        data=output_file,
        evaluators={
            "groundedness": groundedness_evaluator,
        },
        #azure_ai_project=project_scope,
    )
    print(eval_output)


    # %%
    pd.DataFrame(eval_output["rows"])

    # %%
    print(eval_output["metrics"])

    json_result = json.dumps(eval_output, indent=4)

    with Path.open("/model/rai_ground_result.json", "w") as f:
        f.write(json_result)
