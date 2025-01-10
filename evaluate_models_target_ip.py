# %%
#%pip install azure-ai-evaluation --upgrade
#%pip install promptflow-azure --upgrade

# %%
from pprint import pprint
from typing import List, Dict, Any, Optional

import pandas as pd
import random
import json

from azure.ai.evaluation.simulator import AdversarialScenario
from azure.ai.evaluation.simulator import AdversarialSimulator
from azure.identity import AzureCliCredential
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
credential = AzureCliCredential()

# %%
async def callback(
    messages: List[Dict],
    stream: bool = False,
    session_state: Any = None,
) -> dict:
    query = messages["messages"][0]["content"]
    context = None

    # Add file contents for summarization or re-write
    if 'file_content' in messages["template_parameters"]:
        query += messages["template_parameters"]['file_content']
    
    # Call your own endpoint and pass your query as input. Make sure to handle your function_call_to_your_endpoint's error responses.
    model = "onnx-model"
    target=ModelEndpoints(env_var, model)
    response = target(query)["response"]
    
    # Format responses in OpenAI message protocol
    formatted_response = {
        "content": response,
        "role": "assistant",
        "context": {},
    }

    messages["messages"].append(formatted_response)
    return {
        "messages": messages["messages"],
        "stream": stream,
        "session_state": session_state
    }


async def async_main():


    # IP

    scenario = AdversarialScenario.ADVERSARIAL_CONTENT_PROTECTED_MATERIAL
    adversarial_simulator = AdversarialSimulator(azure_ai_project=azure_ai_project, credential=credential)
    #indirect_attack_simulator=IndirectAttackSimulator(azure_ai_project=azure_ai_project)

    outputs = await adversarial_simulator(
            scenario=scenario, # required adversarial scenario to simulate
            target=callback, # callback function to simulate against
            max_conversation_turns=1, #optional, applicable only to conversation scenario
            max_simulation_results=3, #optional
        )

    # By default simulator outputs json, use the following helper function to convert to QA pairs in jsonl format

    print(outputs.to_eval_qr_json_lines())

    # %%
    from pathlib import Path

    with Path.open("outputs_ip.jsonl", "w") as f:
        f.write(outputs.to_eval_qr_json_lines())



    # %%
    filepath = 'outputs_ip.jsonl'
    df = pd.read_json(filepath, lines=True)
    print(df.head())



    # %%
    #from app_target import ModelEndpoints
    import pathlib


    from azure.ai.evaluation import evaluate
    from azure.ai.evaluation import (
        ProtectedMaterialEvaluator,
    )


    #content_safety_evaluator = ContentSafetyEvaluator(azure_ai_project=azure_ai_project, credential=credential)
    ip_evaluator = ProtectedMaterialEvaluator(azure_ai_project=azure_ai_project, credential=credential)


    models = [
        "onnx-model",
    ]

    path = str(pathlib.Path(pathlib.Path.cwd())) + "/outputs_ip.jsonl"

    for model in models:
        randomNum = random.randint(1111, 9999)
        results = evaluate(
            evaluation_name="Eval-Run-" + str(randomNum) + "-" + model.title(),
            data=path,
            evaluators={
                #"content_safety": content_safety_evaluator,
                "ip": ip_evaluator,
            },
            evaluator_config={
                #"content_safety": {"query": "${data.query}", "response": "${data.response}"},
                "ip": {"query": "${data.query}", "response": "${data.response}"},
            },
            #azure_ai_project=azure_ai_project,  # optional to store the evaluation results in Azure AI Studio
        )

    # %%
    pprint(results)

    # %%
    pd.DataFrame(results["rows"])

    # %%
    pprint(results["metrics"])


    json_result = json.dumps(results, indent=4)

    with Path.open("/model/rai_ip_result.json", "w") as f:
        f.write(json_result)
