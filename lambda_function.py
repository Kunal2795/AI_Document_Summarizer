import json
import boto3
import os
import time
import random
from urllib.parse import unquote_plus
import botocore.exceptions

s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

MAX_RETRIES = 5

def invoke_bedrock_model(payload):
    for attempt in range(MAX_RETRIES):
        try:
            return bedrock.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                contentType='application/json',
                accept='application/json',
                body=json.dumps(payload)
            )
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"Throttled. Retry {attempt + 1}, waiting {wait:.2f}s...")
                time.sleep(wait)
            else:
                raise
    raise Exception("Exceeded max retry attempts due to throttling.")

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = unquote_plus(event['Records'][0]['s3']['object']['key'])

    if not key.endswith(".txt") or not key.startswith("uploads/"):
        print(f"Ignored file: {key}")
        return

    # Prevent reprocessing summary files
    if "_summary" in os.path.basename(key):
        print("Skipping already summarized file.")
        return

    # Read file content
    response = s3.get_object(Bucket=bucket, Key=key)
    document_text = response['Body'].read().decode('utf-8')[:10000]

    # Prompt for Claude
    messages = [
        {
            "role": "user",
            "content": f"Summarize the following document:\n\n{document_text}"
        }
    ]

    payload = {
        "messages": messages,
        "max_tokens": 500,
        "anthropic_version": "bedrock-2023-05-31"
    }

    # Bedrock call with retries
    bedrock_response = invoke_bedrock_model(payload)
    result = json.loads(bedrock_response['body'].read())
    summary = result['content'][0]['text']

    # Prepare output path
    base_name = os.path.basename(key).replace(".txt", "")
    summary_key = f"summaries/{base_name}_summary.txt"

    s3.put_object(Bucket=bucket, Key=summary_key, Body=summary.encode("utf-8"))

    return {
        "statusCode": 200,
        "body": f"Summary saved at {summary_key}"
    }
