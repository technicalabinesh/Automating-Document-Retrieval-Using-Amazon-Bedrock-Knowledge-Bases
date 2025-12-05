import boto3
import json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

kb_id = "YOUR_KB_ID"

def query_kb(question):
    payload = {
        "query": question,
        "knowledgeBaseId": kb_id,
        "modelId": "amazon.titan-text-premier-v1:0",
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {"numberOfResults": 5}
        }
    }

    response = bedrock.invoke_model(
        body=json.dumps(payload),
        modelId="amazon.titan-text-premier-v1:0",
        accept="application/json",
        contentType="application/json"
    )

    result = json.loads(response["body"].read())
    return result

print(query_kb("What does this document say about compliance rules?"))

import boto3
import os

s3 = boto3.client("s3")
bucket_name = "YOUR_S3_BUCKET"

def upload_documents(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            s3.upload_file(file_path, bucket_name, f"documents/{file}")
            print(f"Uploaded: {file}")

upload_documents("./docs")

