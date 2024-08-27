import os
import time
from datetime import datetime

import argilla as rg
import requests
from datasets import load_dataset
from distilabel.llms import InferenceEndpointsLLM
from distilabel.steps.tasks import TextGeneration, UltraFeedback

# Environment variables with defaults
API_KEY = os.environ.get("ARGILLA_API_KEY", "argilla.apikey")
API_URL = os.environ.get("ARGILLA_API_URL", "http://localhost:6900")
MAX_RECORDS = int(os.environ.get("MAX_RECORDS", 10))
LLAMA_MODEL_ID = os.environ.get(
    "LLAMA_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct"
)
GEMMA_MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-1.1-7b-it")
ULTRAFEEDBACK_MODEL_ID = os.environ.get(
    "ULTRAFEEDBACK_MODEL_ID", "meta-llama/Meta-Llama-3.1-70B-Instruct"
)

# Initialize Argilla client
client = rg.Argilla(api_key=API_KEY, api_url=API_URL)


def create_dataset():
    return rg.Dataset(
        name=f"triggers_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        settings=rg.Settings(
            fields=[
                rg.TextField("persona"),
                rg.TextField("instruction"),
                rg.TextField("response1"),
                rg.TextField("response2"),
            ],
            questions=[
                rg.LabelQuestion(name="respond", labels=["yes", "no"], required=True),
                rg.TextQuestion(name="improved_instruction", required=False),
                rg.TextQuestion(name="response1_rationale", required=False),
                rg.TextQuestion(name="response2_rationale", required=False),
                rg.RatingQuestion(
                    name="response1_rating", values=[1, 2, 3, 4, 5], required=False
                ),
                rg.RatingQuestion(
                    name="response2_rating", values=[1, 2, 3, 4, 5], required=False
                ),
            ],
        ),
    )


def load_and_upload_records(dataset):
    ds = load_dataset("proj-persona/PersonaHub", "instruction")
    records_to_upload = []
    for sample in ds["train"].to_iterable_dataset():
        record = rg.Record(
            fields={
                "persona": sample["input persona"],
                "instruction": sample["synthesized text"],
                "response1": "",
                "response2": "",
            },
            id=str(hash(sample["synthesized text"])),
        )
        records_to_upload.append(record)
        if len(records_to_upload) == MAX_RECORDS:
            break
    dataset.records.log(records=records_to_upload)


def update_record_fields(record_id, updated_fields):
    url = f"{API_URL}/api/v1/records/{record_id}"
    headers = {
        "accept": "application/json",
        "X-Argilla-Api-Key": API_KEY,
        "Content-Type": "application/json",
    }
    data = {"fields": updated_fields}
    response = requests.patch(url, headers=headers, json=data)
    return response.json()


def delete_response(response_id):
    url = f"{API_URL}/api/v1/responses/{response_id}"
    headers = {
        "accept": "application/json",
        "X-Argilla-Api-Key": API_KEY,
        "Content-Type": "application/json",
    }
    response = requests.delete(url, headers=headers)
    return response.json()


def initialize_text_generation_models():
    llama31 = TextGeneration(
        name="text-generation",
        llm=InferenceEndpointsLLM(
            model_id=LLAMA_MODEL_ID,
            tokenizer_id=LLAMA_MODEL_ID,
        ),
    )
    llama31.load()

    gemma_tiny = TextGeneration(
        name="text-generation",
        llm=InferenceEndpointsLLM(
            model_id=GEMMA_MODEL_ID,
            tokenizer_id=GEMMA_MODEL_ID,
        ),
    )
    gemma_tiny.load()

    return [llama31, gemma_tiny]


def respond_to_record(record: rg.Record, models):
    responses = []
    for task in models:
        print(task.name)
        output = list(task.process([{"instruction": record.fields["instruction"]}]))[0][
            0
        ]
        generation = output["generation"]
        responses.append(generation)
    return responses


def initialize_ultrafeedback():
    ultrafeedback = UltraFeedback(
        aspect="overall-rating",
        llm=InferenceEndpointsLLM(
            model_id=ULTRAFEEDBACK_MODEL_ID,
            tokenizer_id=ULTRAFEEDBACK_MODEL_ID,
        ),
    )
    ultrafeedback.load()
    return ultrafeedback


def add_feedback_suggestions(record, response_1, response_2, ultrafeedback) -> None:
    response = ultrafeedback.process(
        [
            {
                "instruction": "trivia questions",
                "generations": [
                    response_1,
                    response_2,
                ],
            }
        ],
    )
    response = list(response)[0][0]
    ratings = response["ratings"]
    rationales = response["rationales"]

    for n, (rating, rationale) in enumerate(zip(ratings, rationales)):
        record.suggestions.add(
            suggestion=rg.Suggestion(
                question_name=f"response{n+1}_rating",
                value=rating,
            )
        )
        record.suggestions.add(
            suggestion=rg.Suggestion(
                question_name=f"response{n+1}_rationale",
                value=rationale,
            )
        )

    for response in record.responses["respond"]:
        response.status = "draft"
    return record


def respond_to_good_instructions(dataset, models, ultrafeedback) -> None:
    updated_records = []
    for record in dataset.records(
        query=rg.Query(filter=rg.Filter(conditions=[("respond.response", "==", "yes")]))
    ):
        response_1, response_2 = respond_to_record(record=record, models=models)
        updated_fields = dict(record.fields)
        updated_fields["response1"] = response_1
        updated_fields["response2"] = response_2
        update_record_fields(
            record_id=record._server_id,
            updated_fields=updated_fields,
        )
        updated_record = add_feedback_suggestions(
            record=record,
            response_1=response_1,
            response_2=response_2,
            ultrafeedback=ultrafeedback,
        )

        updated_records.append(updated_record)
    dataset.records.log(updated_records)


def get_dataset_progress(dataset_id):
    url = f"{API_URL}/api/v1/datasets/{dataset_id}/progress"
    headers = {
        "accept": "application/json",
        "X-Argilla-Api-Key": API_KEY,
        "Content-Type": "application/json",
    }
    response = requests.get(url, headers=headers)
    return response.json()


def main():
    try:
        workspace = rg.Workspace(name="argilla")
        workspace.create()
    except Exception as e:
        print(f"Workspace already exists: {e}")
    dataset = create_dataset()
    dataset.create()
    load_and_upload_records(dataset)

    models = initialize_text_generation_models()
    ultrafeedback = initialize_ultrafeedback()

    _completed = 0
    while True:
        time.sleep(10)
        dataset_progress = get_dataset_progress(dataset_id=dataset.id)
        completed = dataset_progress["completed"]
        if completed > _completed:
            print(f"Completed {completed} records")
            _completed = completed
            respond_to_good_instructions(dataset, models, ultrafeedback)


if __name__ == "__main__":
    main()
