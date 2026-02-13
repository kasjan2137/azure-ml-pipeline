# Project 2: Azure ML Pipeline — Complete Setup Guide

## 🧒 What Are We Building?

A complete machine learning pipeline that:
1. Takes raw data (CSV file)
2. Cleans and prepares it
3. Trains a model
4. Evaluates how good the model is
5. Registers the model in a catalog
6. Deploys it as an API (anyone can send data and get predictions)

All of this runs in Azure cloud and is tracked/versioned.

---

## PHASE 1: Create Azure ML Workspace (10 minutes)

> **What is an Azure ML Workspace?**
> It's your ML headquarters in the cloud. Everything ML lives here:
> - Your data
> - Your code/experiments
> - Your trained models
> - Your deployments
> Think of it as your ML project's home folder, but in the cloud.

### Step 1: Create the Workspace

1. Go to **https://portal.azure.com**
2. Click the **search bar** at the top
3. Type: **"Machine Learning"**
4. Click **"Azure Machine Learning"**
5. Click **"+ Create"** → **"New workspace"**
6. Fill in:
   - **Subscription**: Your subscription (with credits)
   - **Resource group**: Click **"Create new"** → name it `rg-ml-pipeline`
   - **Workspace name**: `ml-pipeline-workspace`
   - **Region**: `Sweden Central` (or same as Project 1)
7. Click **"Review + create"**
8. Click **"Create"**
9. Wait 2-3 minutes (it creates multiple supporting resources automatically)

> **What gets created automatically?**
> Azure ML creates 4 things behind the scenes:
> - **Storage Account**: Where your data files live
> - **Key Vault**: Where secrets/passwords are stored securely
> - **Application Insights**: Monitoring and logging
> - **Container Registry**: Where Docker containers are stored (for deployment)
> You don't need to touch these directly.

### Step 2: Open Azure ML Studio

1. After deployment completes, click **"Go to resource"**
2. Click **"Launch studio"** (big blue button)
3. A NEW tab opens: **https://ml.azure.com**
4. This is **Azure ML Studio** — where you'll spend most of your time

> **Azure ML Studio is different from Azure Portal!**
> - Azure Portal (portal.azure.com) = Admin control panel for ALL Azure services
> - Azure ML Studio (ml.azure.com) = The ML-specific workspace interface
> Think of Portal as the building lobby and Studio as your office inside.

---

## PHASE 2: Create Compute Resources (5 minutes)

> **What is Compute?**
> A computer in the cloud that runs your code.
> Your laptop could run this too, but cloud compute:
> - Can be more powerful (GPUs)
> - Can scale up/down
> - Runs even when your laptop is off
>
> TWO TYPES:
> - **Compute Instance**: A personal dev machine (like your laptop, but in the cloud)
> - **Compute Cluster**: A group of machines for heavy training jobs (auto-scales)

### Step 3: Create a Compute Instance (for development)

1. In ML Studio, click **"Compute"** in the left sidebar
2. Click the **"Compute instances"** tab
3. Click **"+ New"**
4. Fill in:
   - **Compute name**: `dev-instance`
   - **Virtual machine type**: `CPU`
   - **Virtual machine size**: Click **"Select from all options"**
     - Search for `Standard_DS11_v2` (2 cores, 14GB RAM)
     - This costs ~$0.18/hour — fine for our purposes
5. Click **"Create"**

⚠️ **IMPORTANT**: Set auto-shutdown!
6. Once created, click on `dev-instance`
7. Click **"Schedule"** or look for auto-shutdown settings
8. Enable **"Idle shutdown"** after 30 minutes
9. This prevents you from accidentally burning money overnight

### Step 4: Create a Compute Cluster (for training)

1. Click the **"Compute clusters"** tab
2. Click **"+ New"**
3. Fill in:
   - **Location**: Same as workspace
   - **Virtual machine tier**: `Dedicated`
   - **Virtual machine type**: `CPU`
   - **Virtual machine size**: `Standard_DS3_v2` (4 cores, 14GB RAM)
4. Click **"Next"**
5. Fill in:
   - **Compute name**: `training-cluster`
   - **Minimum number of nodes**: `0` ← CRITICAL! This means it scales to zero when idle
   - **Maximum number of nodes**: `2`
   - **Idle seconds before scale down**: `120` (2 minutes)
6. Click **"Create"**

> **Why minimum nodes = 0?**
> If min = 0, the cluster shuts down completely when not in use.
> You pay NOTHING when it's idle. It just takes 2-3 minutes to spin up.
> If min = 1, you're paying 24/7 even when sleeping. Don't do that.

---

## PHASE 3: Understand What We'll Build in Code

Now we switch to VS Code for the actual ML pipeline code.

The pipeline has these steps:

```
Raw Data (CSV)
     │
     ▼
Step 1: Data Preparation
     │ (clean, split into train/test)
     ▼
Step 2: Training
     │ (train model on training data)
     ▼
Step 3: Evaluation
     │ (test model on test data, compute metrics)
     ▼
Step 4: Registration
     │ (save model in Azure ML Model Registry)
     ▼
Step 5: Deployment
     │ (create API endpoint for predictions)
     ▼
Live API Endpoint
```

All of this is code that runs IN Azure ML, not on your laptop.
Your laptop just SENDS the code to Azure ML.

---

## PHASE 4: Get Your Workspace Credentials

### Step 5: Note These Values

In Azure ML Studio, you need:

1. **Subscription ID**: 
   - Azure Portal → Search "Subscriptions" → Copy the ID
2. **Resource Group**: `rg-ml-pipeline`
3. **Workspace Name**: `ml-pipeline-workspace`

These go in your code's config file.

### Step 6: Install Azure CLI (on your laptop)

We use the Azure CLI to authenticate your code with Azure.

```bash
# macOS
brew install azure-cli

# Or download from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# Then login:
az login
# This opens a browser window — sign in with your Azure account
```

After `az login`, your code can automatically connect to Azure ML.
No API keys needed! (Azure CLI handles authentication.)

---

## Next: Write the Code in VS Code

Now go to the Python files in this project.
Follow the numbered order: 01, 02, 03, etc.
