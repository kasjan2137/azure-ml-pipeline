"""
Azure ML Configuration.

This file connects your local code to your Azure ML workspace.

HOW AUTHENTICATION WORKS:
1. You run 'az login' on your terminal
2. This stores a token on your computer
3. When your code runs, it uses that token automatically
4. No API keys needed! (This is called "DefaultAzureCredential")

TO FIND YOUR VALUES:
- subscription_id: Azure Portal → Subscriptions → copy the ID
- resource_group: Whatever you named it (e.g., "rg-ml-pipeline")
- workspace_name: Whatever you named it (e.g., "ml-pipeline-workspace")
"""

import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from dotenv import load_dotenv

load_dotenv()

# ---- YOUR AZURE ML WORKSPACE DETAILS ----
# Fill these in with YOUR values from Azure Portal

SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "paste-your-subscription-id-here")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "rg-ml-pipeline")
WORKSPACE_NAME = os.getenv("AZURE_ML_WORKSPACE", "ml-pipeline-workspace")

# Compute names (must match what you created in Azure ML Studio)
COMPUTE_CLUSTER = "training-cluster"
COMPUTE_INSTANCE = "dev-instance"


def get_ml_client() -> MLClient:
    """
    Create connection to Azure ML workspace.
    
    MLClient is the main object for interacting with Azure ML.
    Think of it as your "remote control" for the workspace.
    
    With it you can:
    - Submit training jobs
    - Register models
    - Create endpoints
    - Manage data
    """
    credential = DefaultAzureCredential()
    
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )
    
    return ml_client


if __name__ == "__main__":
    print("Testing Azure ML connection...")
    
    try:
        client = get_ml_client()
        ws = client.workspaces.get(WORKSPACE_NAME)
        print(f"✅ Connected to workspace: {ws.name}")
        print(f"   Location: {ws.location}")
        print(f"   Resource group: {RESOURCE_GROUP}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n💡 Make sure you've run 'az login' first!")
        print("💡 Check that SUBSCRIPTION_ID, RESOURCE_GROUP, and WORKSPACE_NAME are correct")
