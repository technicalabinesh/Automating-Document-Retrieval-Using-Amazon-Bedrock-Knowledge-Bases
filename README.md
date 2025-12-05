# bedrock-knowledge-base-retrieval

This document contains the full project files for **Workshop 2 — Retrieve Data from Documents with Amazon Bedrock & Knowledge Bases** (Titan Text Premier example). Copy each file into the same structure before pushing to GitHub.

---

## Project structure

```
bedrock-knowledge-base-retrieval/
├── .kiro/
│   └── config.json
├── src/
│   ├── upload_documents.py
│   ├── query_kb.py
│   └── utils.py
├── requirements.txt
├── README.md
└── example_output.json
```

---

## .kiro/config.json

```json
{
  "project": "bedrock-knowledge-base-retrieval",
  "description": "Kiro placeholder config for Kiro Week 2 submission",
  "author": "ABINESH M",
  "created": "2025-12-05",
  "notes": "This is a minimal placeholder required by the challenge. Replace with your Kiro workflow files if you used Kiro in AWS."
}
```

---

## requirements.txt

```
boto3>=1.28.0
botocore>=1.31.0
faiss-cpu==1.7.4
tqdm
python-magic
pdfminer.six
python-docx
numpy
scikit-learn
```

> Note: `faiss-cpu` binary version may vary by platform — if it fails to install on Windows, use an alternative (Annoy, Milvus, or a hosted vector DB).

---

## src/utils.py

```python
"""
Utilities: document loaders and simple text splitting.
"""
import os
import json
import math
from pathlib import Path
from pdfminer.high_level import extract_text
from docx import Document


def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(path: str) -> str:
    return extract_text(path)


def load_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_document(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext in [".docx"]:
        return load_docx(path)
    if ext in [".txt", ".md"]:
        return load_txt(path)
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, max_tokens: int = 800) -> list:
    """A simple chunker by characters — not token-accurate but practical.
    Splits text into chunks of roughly `max_tokens` characters.
    """
    if not text:
        return []
    size = max_tokens * 4  # heuristic: 1 token ~ 4 chars
    chunks = [text[i:i+size].strip() for i in range(0, len(text), size) if text[i:i+size].strip()]
    return chunks


if __name__ == "__main__":
    print("utils module loaded")
```

---

## src/upload_documents.py

```python
"""
Scan a directory, extract text from documents, create embeddings via Amazon Bedrock (Titan embeddings or Titan Text), and store vectors in a local FAISS index.

This script is intentionally conservative: Bedrock input/output formats may vary by model and AWS SDK version.

Usage:
    python src/upload_documents.py --docs ./documents --index_path ./vectors.index --meta ./metadata.json

"""
import argparse
import json
import os
import time
from pathlib import Path

import boto3
import numpy as np
import faiss
from tqdm import tqdm

from utils import load_document, chunk_text


def init_bedrock(region_name: str = "us-east-1"):
    client = boto3.client("bedrock-runtime", region_name=region_name)
    return client


def embed_text(bedrock_client, model_id: str, text: str):
    # This wrapper uses InvokeModel. Output parsing may need tweaking depending on the model's response format.
    response = bedrock_client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"input": text})
    )

    # `body` is a streaming HTTPResponse-like object. boto3 returns bytes in body.
    body_bytes = response['body'].read()
    payload = json.loads(body_bytes.decode('utf-8'))

    # Expected: payload["outputs"][0]["embedding"] or payload["embedding"] depending on model.
    # Try common locations safely.
    emb = None
    if isinstance(payload, dict):
        if "outputs" in payload and isinstance(payload["outputs"], list):
            out0 = payload["outputs"][0]
            if isinstance(out0, dict) and "embedding" in out0:
                emb = out0["embedding"]
        if emb is None and "embedding" in payload:
            emb = payload["embedding"]

    if emb is None:
        raise ValueError("Unable to parse embedding from Bedrock response. Inspect raw response: " + json.dumps(payload)[:1000])

    return np.array(emb, dtype=np.float32)


def main(docs_dir, index_path, meta_path, model_id, region):
    bedrock = init_bedrock(region)

    paths = [p for p in Path(docs_dir).glob("**/*") if p.is_file()]
    print(f"Found {len(paths)} files to index")

    embeddings = []
    metadatas = []

    for p in tqdm(paths):
        try:
            text = load_document(str(p))
        except Exception as e:
            print(f"Skipping {p}: {e}")
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            emb = embed_text(bedrock, model_id, chunk)
            embeddings.append(emb)
            metadatas.append({"source": str(p), "chunk_id": i, "text_snippet": chunk[:200]})
            time.sleep(0.1)  # be gentle with rate limits

    if not embeddings:
        print("No embeddings created — exiting")
        return

    d = len(embeddings[0])
    xb = np.vstack(embeddings).astype('float32')

    index = faiss.IndexFlatL2(d)
    index.add(xb)
    faiss.write_index(index, index_path)

    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

    print(f"Wrote index {index_path} with {len(embeddings)} vectors")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--docs', required=True)
    p.add_argument('--index_path', default='./vectors.index')
    p.add_argument('--meta', default='./metadata.json')
    p.add_argument('--model_id', default='amazon.titan-text-embed', help='Bedrock model id for embeddings')
    p.add_argument('--region', default='us-east-1')
    args = p.parse_args()
    main(args.docs, args.index_path, args.meta, args.model_id, args.region)
```

---

## src/query_kb.py

```python
"""
Load local FAISS index + metadata, run a similarity search for a query, then ask Titan Text Premier to answer the question using retrieved context.

Usage:
    python src/query_kb.py --index ./vectors.index --meta ./metadata.json --query "What is the refund policy?"

"""
import argparse
import json
from pathlib import Path

import boto3
import faiss
import numpy as np


def init_bedrock(region_name: str = "us-east-1"):
    return boto3.client("bedrock-runtime", region_name=region_name)


def load_index(index_path: str):
    if not Path(index_path).exists():
        raise FileNotFoundError(index_path)
    return faiss.read_index(index_path)


def embed_query(bedrock_client, model_id: str, query: str):
    response = bedrock_client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"input": query})
    )
    body_bytes = response['body'].read()
    payload = json.loads(body_bytes.decode('utf-8'))

    emb = None
    if isinstance(payload, dict):
        if "outputs" in payload and isinstance(payload["outputs"], list):
            out0 = payload["outputs"][0]
            if isinstance(out0, dict) and "embedding" in out0:
                emb = out0["embedding"]
        if emb is None and "embedding" in payload:
            emb = payload["embedding"]

    if emb is None:
        raise ValueError("Unable to parse embedding from Bedrock response for query")

    return np.array(emb, dtype=np.float32)


def run_similarity(index, emb, top_k=5):
    emb = emb.reshape(1, -1).astype('float32')
    D, I = index.search(emb, top_k)
    return I[0].tolist(), D[0].tolist()


def answer_with_context(bedrock_client, model_id: str, query: str, contexts: list):
    # Compose a prompt that includes retrieved context
    prompt = """
You are an assistant. Use the following retrieved document snippets to answer the question. Be concise and cite the source file paths.

CONTEXT:
"""
    for c in contexts:
        prompt += f"- Source: {c['source']}\n  Snippet: {c['text_snippet']}\n\n"
    prompt += f"QUESTION: {query}\nANSWER:"

    response = bedrock_client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"input": prompt, "max_tokens": 512})
    )
    body_bytes = response['body'].read()
    payload = json.loads(body_bytes.decode('utf-8'))

    # Try to extract textual answer
    answer = None
    if isinstance(payload, dict):
        if "outputs" in payload and isinstance(payload["outputs"], list):
            out0 = payload["outputs"][0]
            if isinstance(out0, dict) and "text" in out0:
                answer = out0["text"]
        if answer is None and "output" in payload:
            answer = payload.get("output")

    if answer is None:
        # Fallback: stringify payload
        answer = json.dumps(payload)[:2000]

    return answer


def main(index_path, meta_path, query, embed_model_id, answer_model_id, region):
    bedrock = init_bedrock(region)
    index = load_index(index_path)

    with open(meta_path, 'r', encoding='utf-8') as f:
        metas = json.load(f)

    q_emb = embed_query(bedrock, embed_model_id, query)
    ids, dists = run_similarity(index, q_emb, top_k=5)
    contexts = [metas[i] for i in ids]

    answer = answer_with_context(bedrock, answer_model_id, query, contexts)

    print("--- Retrieved snippets ---")
    for i, c in enumerate(contexts):
        print(i+1, c['source'], c['text_snippet'][:200])

    print("\n--- Answer ---\n")
    print(answer)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--index', required=True)
    p.add_argument('--meta', required=True)
    p.add_argument('--query', required=True)
    p.add_argument('--embed_model', default='amazon.titan-text-embed')
    p.add_argument('--answer_model', default='amazon.titan-text-premier')
    p.add_argument('--region', default='us-east-1')
    args = p.parse_args()
    main(args.index, args.meta, args.query, args.embed_model, args.answer_model, args.region)
```

---

## example_output.json

```json
{
  "query": "What is the refund policy?",
  "retrieved": [
    {"source": "documents/terms.pdf", "snippet": "Refunds are issued within 30 days of purchase..."}
  ],
  "answer": "According to the documents, refunds are issued within 30 days of purchase. For details see documents/terms.pdf"
}
```

---

## README.md

````markdown
# Bedrock Knowledge Base Retrieval

Workshop 2 — Retrieve Data from Documents with Amazon Bedrock & Knowledge Bases (Titan Text Premier)

## What this project does

This repository demonstrates a minimal pipeline to:

1. Extract text from documents (PDF, DOCX, TXT).
2. Chunk documents and create embeddings using Amazon Bedrock.
3. Store embeddings in a local FAISS index with metadata.
4. Run a similarity search for a user query and use Titan Text Premier to answer using retrieved context.

> NOTE: This is a reference/template project for the workshop. Bedrock's exact input/output formats and model IDs may vary; inspect raw responses when running.

## Files

- `.kiro/config.json` — placeholder Kiro config (required for Kiro Week 2 submission).
- `src/upload_documents.py` — load documents, create embeddings, build FAISS index.
- `src/query_kb.py` — query the index and ask Titan Text Premier to answer with context.
- `src/utils.py` — document loaders + chunker.
- `requirements.txt` — pip dependencies.

## Quick start

1. Create an AWS profile with Bedrock access and ensure your AWS account is enabled for Bedrock.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
````

3. Prepare a `./documents` folder with PDFs, DOCX, or TXT files.
4. Build the index:

```bash
python src/upload_documents.py --docs ./documents --index_path ./vectors.index --meta ./metadata.json --model_id amazon.titan-text-embed --region us-east-1
```

5. Query the KB:

```bash
python src/query_kb.py --index ./vectors.index --meta ./metadata.json --query "How do I request a refund?" --embed_model amazon.titan-text-embed --answer_model amazon.titan-text-premier --region us-east-1
```

## Caveats & Notes

* The script expects Bedrock `invoke_model` outputs to include embeddings under common keys — different models may return different JSON shapes. If your model returns embeddings in a different field, update the parsing code in `upload_documents.py` and `query_kb.py`.
* For production, replace FAISS with a scalable vector DB (Milvus, Pinecone, OpenSearch, etc.).
* Ensure IAM permissions include `bedrock:InvokeModel`.

## How Kiro helped (for your blog)

* Use screenshots of Kiro generating code snippets and running the workflow in AWS Builder Center. Include a 30–60s recording of the pipeline running.

## License

MIT

```

---


Thank you — all files are included above. Copy them into your project folders and push to GitHub. If you want, I can also:

- generate a ready-to-upload ZIP, or
- export each file separately, or
- customize the README with your name and specific document examples.

Which one do you want next?

```
