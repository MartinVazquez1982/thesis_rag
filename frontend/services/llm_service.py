import os
import requests

URL = f"{os.getenv("RAG_URL")}/query"

def get_llm_response(question: str) -> str:
    payload = {
        "question": question
    }
    try:
        response = requests.post(URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["answer"]
    except requests.exceptions.RequestException as e:
        pass

