# Task 65 — Microsoft Azure: AI & ML Services
# Reference: Microsoft Learn — https://learn.microsoft.com/en-us/training/

"""
This task covers Microsoft Azure video lectures from Microsoft Learning.
Below is a structured summary of key Azure AI/ML services and concepts.
"""

print("=" * 60)
print("Task 65 — Microsoft Azure: AI & ML Services")
print("=" * 60)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1. WHAT IS MICROSOFT AZURE?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Microsoft Azure is a cloud computing platform offering 200+
services including computing, storage, networking, databases,
AI, and machine learning tools.

Key categories:
  • Compute        → Virtual Machines, Azure Functions
  • Storage        → Blob Storage, Data Lake
  • Databases      → Azure SQL, Cosmos DB
  • AI/ML          → Azure ML, Cognitive Services, OpenAI
  • Networking     → Virtual Network, Load Balancer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 2. AZURE MACHINE LEARNING (Azure ML)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Azure ML is an end-to-end platform for building, training,
deploying, and monitoring ML models.

Core components:
  • Workspace       → central hub for all ML assets
  • Compute Cluster → scalable VMs for training
  • Datastores      → connected data sources
  • Datasets        → versioned, reusable data
  • Pipelines       → automated ML workflows
  • Designer        → drag-and-drop ML builder (no-code)
  • AutoML          → automatic algorithm selection

Workflow:
  Data → Prepare → Train → Evaluate → Deploy → Monitor

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 3. AZURE COGNITIVE SERVICES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pre-built AI APIs — no ML expertise required:

  Vision:
    • Computer Vision    → image analysis, OCR
    • Face API           → face detection/recognition
    • Custom Vision      → train image classifiers

  Language:
    • Text Analytics     → sentiment, key phrases, entities
    • Translator         → 100+ language translation
    • LUIS               → Language Understanding
    • QnA Maker          → FAQ chatbot builder

  Speech:
    • Speech-to-Text     → voice transcription
    • Text-to-Speech     → voice synthesis
    • Speaker Recognition

  Decision:
    • Anomaly Detector   → time series anomaly detection
    • Personalizer       → content recommendation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 4. AZURE OPENAI SERVICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Provides access to OpenAI models (GPT-4, DALL-E, Whisper)
through Azure's secure infrastructure.

Use cases:
  • Text generation and summarisation
  • Code generation (GitHub Copilot backend)
  • Image generation with DALL-E
  • Embeddings for semantic search
  • RAG (Retrieval Augmented Generation) pipelines

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 5. AZURE DATA SERVICES FOR ML
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Azure Data Factory   → ETL pipelines, data movement
  • Azure Databricks     → Apache Spark for big data ML
  • Azure Synapse        → analytics + data warehouse
  • Azure Data Lake      → storing raw/unstructured data

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 6. AZURE ML — PYTHON SDK EXAMPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

azure_code_example = '''
# Example: Connecting to Azure ML Workspace (Python SDK v2)
# pip install azure-ai-ml azure-identity

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Authenticate and connect
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="your-subscription-id",
    resource_group_name="your-resource-group",
    workspace_name="your-workspace-name"
)

# List available compute targets
for compute in ml_client.compute.list():
    print(compute.name, compute.type)

# Submit a training job
from azure.ai.ml import command
from azure.ai.ml.entities import Environment

job = command(
    code="./src",
    command="python train.py --learning_rate 0.01 --epochs 20",
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
    compute="my-cluster",
    display_name="sklearn-training-job"
)
returned_job = ml_client.jobs.create_or_update(job)
print("Job submitted:", returned_job.name)
'''

print(azure_code_example)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 7. KEY CERTIFICATIONS (FREE via Microsoft Learn)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • AI-900 : Azure AI Fundamentals       ← Start here
  • DP-100 : Azure Data Scientist Assoc. ← Advanced ML
  • AI-102 : Azure AI Engineer Assoc.    ← NLP/Vision apps

  Microsoft Learn path (free):
  https://learn.microsoft.com/en-us/training/paths/get-started-with-artificial-intelligence-on-azure/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 8. AZURE vs AWS vs GCP — QUICK COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Feature          Azure ML      AWS SageMaker    GCP Vertex AI
  ─────────────────────────────────────────────────────────────
  AutoML           ✓             ✓                ✓
  No-code Builder  Designer      Canvas           AutoML Tables
  LLM Access       Azure OpenAI  Bedrock          Vertex AI LLMs
  Free Tier        ✓ (limited)   ✓ (limited)      ✓ (limited)
  Best for         Enterprise    AWS ecosystem    Research/Data

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 NOTE: To complete this task, watch the official video lectures
 at: https://learn.microsoft.com/en-us/training/
 and take screenshots of completed modules as GitHub evidence.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
