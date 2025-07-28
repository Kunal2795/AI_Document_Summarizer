import json
import boto3
import os
import io
from urllib.parse import unquote_plus

# External modules
from pdfminer.high_level import extract_text  # replacing fitz
import docx  # python-docx

s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')


def extract_text_from_pdf(file_stream):
    temp_pdf_path = "/tmp/temp_input.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(file_stream.read())
    return extract_text(temp_pdf_path)


def extract_text_from_docx(file_stream):
    document = docx.Document(file_stream)
    return '\n'.join([para.text for para in document.paragraphs])


def lambda_handler(event, context):
    try:
        # Parse S3 event
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = unquote_plus(event['Records'][0]['s3']['object']['key'])

        # Validate source folder
        if not key.startswith("uploads/"):
            print(f"Skipped key not in uploads/: {key}")
            return {"statusCode": 400, "body": "File not in uploads/ folder"}

        print(f"Received file: s3://{bucket}/{key}")
        file_ext = key.lower().split('.')[-1]

        # Get file object
        s3_object = s3.get_object(Bucket=bucket, Key=key)
        file_stream = io.BytesIO(s3_object['Body'].read())

        # Extract text based on file type
        if file_ext == "txt":
            document_text = file_stream.read().decode("utf-8")
        elif file_ext == "pdf":
            document_text = extract_text_from_pdf(file_stream)
        elif file_ext in ["docx", "doc"]:
            document_text = extract_text_from_docx(file_stream)
        else:
            print(f"Unsupported file type: {key}")
            return {"statusCode": 400, "body": "Unsupported file type"}

        print(f"Document length: {len(document_text)} characters")

        # Claude 3 Messages API format
        messages = [
            {
                "role": "user",
                "content": f"Summarize the following document:\n\n{document_text}"
            }
        ]

        # Call Claude 3 via Bedrock (Messages API)
        bedrock_response = bedrock.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": 500
            })
        )

        result = json.loads(bedrock_response['body'].read())
        summary = result['content'][0]['text']
        print("Summary received from Claude.")

        # Save to S3
        filename = os.path.basename(key).rsplit('.', 1)[0] + "_summary.txt"
        summary_key = f"summaries/{filename}"
        s3.put_object(Bucket=bucket, Key=summary_key, Body=summary.encode("utf-8"))
        print(f"Summary saved to: s3://{bucket}/{summary_key}")

        return {
            "statusCode": 200,
            "body": f"Summary saved at s3://{bucket}/{summary_key}"
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": f"Error: {str(e)}"
        }
