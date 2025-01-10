import requests
from typing_extensions import Self
from typing import TypedDict
from promptflow.tracing import trace


class ModelEndpoints:
    def __init__(self: Self, env: dict, model_type: str) -> str:
        self.env = env
        self.model_type = model_type

    class Response(TypedDict):
        query: str
        response: str

    @trace
    def __call__(self: Self, question: str) -> Response:
        #self.model_type == "onnx-model":
        output = self.call_onnx_endpoint(question)
        return output

    def query(self: Self, endpoint: str, headers: str, payload: str) -> str:
        response = requests.post(url=endpoint, headers=headers, json=payload)
        return response.json()

    def call_onnx_endpoint(self: Self, question: str) -> Response:
        endpoint = self.env["onnx-model"]["endpoint"]
        key = self.env["onnx-model"]["key"]

        headers = {"Content-Type": "application/json", "api-key": key}
        #headers = {"Content-Type": "application/json"}

        #payload = {"messages": [{"role": "user", "content": question}], "max_tokens": 500}

        payload = {"text": question}

        output = self.query(endpoint=endpoint, headers=headers, payload=payload)
        #answer = output["choices"][0]["message"]["content"]
        answer = output["response"]
        return {"query": question, "response": answer}