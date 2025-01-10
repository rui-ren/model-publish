# %%
from pprint import pprint
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import pathlib
import random
import json

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

async def async_main_eci():
    # ECI
    from azure.ai.evaluation.simulator import AdversarialScenario
    from azure.ai.evaluation.simulator._adversarial_scenario import _UnstableAdversarialScenario

    scenario = _UnstableAdversarialScenario.ECI
    adversarial_simulator = AdversarialSimulator(azure_ai_project=azure_ai_project, credential=credential)


    outputs = await adversarial_simulator(
            scenario=scenario, # required adversarial scenario to simulate
            target=callback, # callback function to simulate against
            max_conversation_turns=1, #optional, applicable only to conversation scenario
            max_simulation_results=3, #optional
        )

    # By default simulator outputs json, use the following helper function to convert to QA pairs in jsonl format
    #outputs.to_eval_qa_json_lines()
    print(outputs.to_eval_qr_json_lines())

    # %%
    with Path.open("outputs_eci.jsonl", "w") as f:
        f.write(outputs.to_eval_qr_json_lines())


    # %%
    filepath = 'outputs_eci.jsonl'
    df = pd.read_json(filepath, lines=True)
    print(df.head())



    # %%
    #from app_target import ModelEndpoints

    from azure.ai.evaluation import evaluate
    from azure.ai.evaluation._evaluators._eci._eci import ECIEvaluator


    #content_safety_evaluator = ContentSafetyEvaluator(azure_ai_project=azure_ai_project, credential=credential)
    eci_evaluator = ECIEvaluator(azure_ai_project=azure_ai_project,credential=credential)


    models = [
        "onnx-model",
    ]

    path = str(pathlib.Path(pathlib.Path.cwd())) + "/outputs_eci.jsonl"

    for model in models:
        randomNum = random.randint(1111, 9999)
        results = evaluate(
            evaluation_name="Eval-Run-" + str(randomNum) + "-" + model.title(),
            data=path,
            #target=ModelEndpoints(env_var, model),
            evaluators={
                #"content_safety": content_safety_evaluator,
                "eci": eci_evaluator,
            },
            evaluator_config={
                #"content_safety": {"query": "${data.query}", "response": "${data.response}"},
                "eci": {"query": "${data.query}", "response": "${data.response}"},
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

    with Path.open("/model/rai_eci_result.json", "w") as f:
        f.write(json_result)
