"""
Azure ML Pipeline Orchestrator.
THIS IS THE MAIN FILE YOU RUN FROM YOUR LAPTOP.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from azure.ai.ml import MLClient, Input, Output, command, dsl
from azure.ai.ml.entities import (
    Environment,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
    Model,
    Data,
)
from azure.ai.ml.constants import AssetTypes
from src.config import get_ml_client, COMPUTE_CLUSTER


# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------
def create_environment(ml_client: MLClient) -> Environment:
    env = Environment(
        name="heart-disease-env",
        description="Environment for heart disease ML pipeline",
        conda_file={
            "name": "ml-pipeline-env",
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.10",
                "pip",
                {
                    "pip": [
                        "scikit-learn",
                        "pandas",
                        "numpy",
                        "mlflow",
                        "azureml-mlflow",
                        "joblib",
                    ]
                },
            ],
        },
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )

    registered_env = ml_client.environments.create_or_update(env)
    print(
        f"✅ Environment '{registered_env.name}' version {registered_env.version} registered"
    )
    return registered_env


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
def upload_dataset(ml_client: MLClient) -> Input:
    """Upload heart disease dataset to Azure ML."""

    data_path = Path(__file__).parent.parent / "data" / "heart_disease.csv"

    if not data_path.exists():
        # Create sample data if it doesn't exist
        print(f"📁 Creating sample dataset at {data_path}")
        data_path.parent.mkdir(parents=True, exist_ok=True)

        import pandas as pd
        import numpy as np

        np.random.seed(42)
        n_samples = 1000

        data = pd.DataFrame(
            {
                "age": np.random.randint(29, 77, n_samples),
                "sex": np.random.randint(0, 2, n_samples),
                "cp": np.random.randint(0, 4, n_samples),
                "trestbps": np.random.randint(94, 200, n_samples),
                "chol": np.random.randint(126, 564, n_samples),
                "fbs": np.random.randint(0, 2, n_samples),
                "restecg": np.random.randint(0, 3, n_samples),
                "thalach": np.random.randint(71, 202, n_samples),
                "exang": np.random.randint(0, 2, n_samples),
                "oldpeak": np.round(np.random.uniform(0, 6.2, n_samples), 1),
                "slope": np.random.randint(0, 3, n_samples),
                "ca": np.random.randint(0, 5, n_samples),
                "thal": np.random.randint(0, 4, n_samples),
                "target": np.random.randint(0, 2, n_samples),
            }
        )
        data.to_csv(data_path, index=False)

    print(f"📂 Using dataset: {data_path}")

    # Register as data asset
    data_asset = Data(
        name="heart-disease-data",
        path=str(data_path),
        type=AssetTypes.URI_FILE,
        description="Heart disease prediction dataset",
    )

    registered_data = ml_client.data.create_or_update(data_asset)
    print(f"✅ Dataset uploaded: {registered_data.name} v{registered_data.version}")

    return Input(
        type=AssetTypes.URI_FILE,
        path=f"azureml:{registered_data.name}:{registered_data.version}",
    )


# -------------------------------------------------------------------
# Pipeline Definition
# -------------------------------------------------------------------
@dsl.pipeline(
    name="heart_disease_pipeline",
    description="End-to-end ML pipeline for heart disease prediction",
    compute=COMPUTE_CLUSTER,
)
def heart_disease_pipeline(input_data: Input):
    """
    Pipeline with 4 steps:
    1. Data Preparation - Clean and split data
    2. Training - Train Random Forest model
    3. Evaluation - Evaluate on test set
    4. Registration - Prepare model for registration
    """

    components_dir = Path(__file__).parent.parent / "components"
    env_name = "heart-disease-env@latest"

    # ---- Step 1: Data Preparation ----
    data_prep_component = command(
        name="data_preparation",
        display_name="1. Data Preparation",
        description="Clean data and split into train/test",
        command=(
            "python 01_data_prep.py "
            "--input_data ${{inputs.raw_data}} "
            "--train_data ${{outputs.train_data}} "
            "--test_data ${{outputs.test_data}} "
            "--test_size 0.2"
        ),
        code=str(components_dir),
        environment=env_name,
        inputs={"raw_data": Input(type="uri_file")},
        outputs={
            "train_data": Output(type="uri_folder"),
            "test_data": Output(type="uri_folder"),
        },
    )
    data_prep_step = data_prep_component(raw_data=input_data)

    # ---- Step 2: Training ----
    train_component = command(
        name="model_training",
        display_name="2. Model Training",
        description="Train Random Forest classifier",
        command=(
            "python 02_train.py "
            "--train_data ${{inputs.train_data}} "
            "--model_output ${{outputs.model_output}} "
            "--n_estimators 100 "
            "--max_depth 10"
        ),
        code=str(components_dir),
        environment=env_name,
        inputs={"train_data": Input(type="uri_folder")},
        outputs={
            "model_output": Output(type="uri_folder"),
        },
    )
    train_step = train_component(train_data=data_prep_step.outputs.train_data)

    # ---- Step 3: Evaluation ----
    eval_component = command(
        name="model_evaluation",
        display_name="3. Model Evaluation",
        description="Evaluate model on test data",
        command=(
            "python 03_evaluate.py "
            "--model_input ${{inputs.model_input}} "
            "--test_data ${{inputs.test_data}} "
            "--evaluation_output ${{outputs.evaluation_output}}"
        ),
        code=str(components_dir),
        environment=env_name,
        inputs={
            "model_input": Input(type="uri_folder"),
            "test_data": Input(type="uri_folder"),
        },
        outputs={
            "evaluation_output": Output(type="uri_folder"),
        },
    )
    eval_step = eval_component(
        model_input=train_step.outputs.model_output,
        test_data=data_prep_step.outputs.test_data,
    )

    # ---- Step 4: Registration ----
    register_component = command(
        name="model_registration",
        display_name="4. Model Registration",
        description="Prepare model for registration",
        command=(
            "python 04_register.py "
            "--model_input ${{inputs.model_input}} "
            "--evaluation_input ${{inputs.evaluation_input}} "
            "--model_name heart-disease-classifier"
        ),
        code=str(components_dir),
        environment=env_name,
        inputs={
            "model_input": Input(type="uri_folder"),
            "evaluation_input": Input(type="uri_folder"),
        },
    )
    register_step = register_component(
        model_input=train_step.outputs.model_output,
        evaluation_input=eval_step.outputs.evaluation_output,
    )

    return {
        "model": train_step.outputs.model_output,
        "evaluation": eval_step.outputs.evaluation_output,
    }


# -------------------------------------------------------------------
# Run Pipeline
# -------------------------------------------------------------------
def run_pipeline():
    """Submit the pipeline to Azure ML."""

    print("=" * 60)
    print("AZURE ML PIPELINE - Heart Disease Prediction")
    print("=" * 60)

    # Connect to workspace
    print("\n🔗 Connecting to Azure ML workspace...")
    ml_client = get_ml_client()
    print("   ✅ Connected!")

    # Create environment
    print("\n📦 Setting up environment...")
    create_environment(ml_client)

    # Upload dataset
    print("\n📊 Preparing dataset...")
    input_data = upload_dataset(ml_client)

    # Create pipeline
    print("\n🔧 Building pipeline...")
    pipeline_job = heart_disease_pipeline(input_data=input_data)

    # Set pipeline-level settings
    pipeline_job.settings.default_compute = COMPUTE_CLUSTER

    # Submit pipeline
    print("\n🚀 Submitting pipeline to Azure ML...")
    job = ml_client.jobs.create_or_update(pipeline_job)

    print("\n" + "=" * 60)
    print("✅ Pipeline submitted!")
    print(f"   Job name: {job.name}")
    print(f"   Status: {job.status}")
    print("\n📺 Watch it run:")
    print(f"   {job.studio_url}")
    print("=" * 60)

    return job


# -------------------------------------------------------------------
# Deploy Model
# -------------------------------------------------------------------
def deploy_model(
    model_name: str = "heart-disease-classifier",
    endpoint_name: str = "heart-disease-endpoint",
):
    """Deploy the model as a managed online endpoint."""

    ml_client = get_ml_client()

    print("\n🚀 DEPLOYING MODEL")
    print("=" * 60)

    # First, register the model from local components folder
    print(f"\n📦 Registering model: {model_name}")

    components_dir = Path(__file__).parent.parent / "components"

    # Check if we have a trained model locally (from a previous run)
    # If not, we'll create a simple model for demonstration
    local_model_dir = Path(__file__).parent.parent / "trained_model"

    if not local_model_dir.exists():
        print("   Creating demo model for deployment...")
        local_model_dir.mkdir(parents=True, exist_ok=True)

        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        # Create a simple trained model
        np.random.seed(42)
        X = np.random.randn(100, 13)
        y = np.random.randint(0, 2, 100)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_scaled, y)

        joblib.dump(model, local_model_dir / "model.pkl")
        joblib.dump(scaler, local_model_dir / "scaler.pkl")

        features = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]
        with open(local_model_dir / "features.txt", "w") as f:
            f.write("\n".join(features))

        print("   ✅ Demo model created")

    # Register model in Azure ML
    model = Model(
        name=model_name,
        path=str(local_model_dir),
        type=AssetTypes.CUSTOM_MODEL,
        description="Heart disease prediction model",
    )

    registered_model = ml_client.models.create_or_update(model)
    print(
        f"   ✅ Model registered: {registered_model.name} v{registered_model.version}"
    )

    # Create endpoint
    print(f"\n🌐 Creating endpoint: {endpoint_name}")
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Heart disease prediction endpoint",
        auth_mode="key",
    )

    try:
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print("   ✅ Endpoint created")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("   ℹ️ Endpoint already exists, reusing")
        else:
            raise

    # Create deployment
    print(f"\n📦 Creating deployment (this takes ~8-10 minutes)...")

    deployment = ManagedOnlineDeployment(
        name="default",
        endpoint_name=endpoint_name,
        model=f"azureml:{registered_model.name}:{registered_model.version}",
        code_configuration=CodeConfiguration(
            code=str(components_dir),
            scoring_script="score.py",
        ),
        environment="heart-disease-env@latest",
        instance_type="Standard_DS2_v2",
        instance_count=1,
    )

    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print("   ✅ Deployment created")

    # Set traffic to 100%
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {"default": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    # Get endpoint details
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    print("\n" + "=" * 60)
    print("✅ DEPLOYMENT COMPLETE!")
    print(f"   Endpoint: {endpoint_name}")
    print(f"   Scoring URL: {endpoint.scoring_uri}")
    print("=" * 60)


# -------------------------------------------------------------------
# Test Endpoint
# -------------------------------------------------------------------
def test_endpoint(endpoint_name: str = "heart-disease-endpoint"):
    """Test the deployed endpoint with sample data."""
    import json
    import tempfile
    import os

    ml_client = get_ml_client()

    # Sample patient data
    sample_data = {
        "input_data": {
            "columns": [
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal",
            ],
            "data": [
                [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1],
                [37, 0, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2],
            ],
        }
    }

    print(f"\n🧪 Testing endpoint: {endpoint_name}")
    print(f"   Sending 2 patient records...")

    # Write test data to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_data, f)
        temp_path = f.name

    try:
        response = ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            request_file=temp_path,
        )

        result = json.loads(response)
        print(f"\n📊 Predictions:")

        if "predictions" in result:
            for i, pred in enumerate(result["predictions"]):
                risk = "HIGH RISK" if pred == 1 else "LOW RISK"
                print(f"   Patient {i+1}: {risk}")
        else:
            print(f"   Response: {result}")

    finally:
        os.unlink(temp_path)


# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------
def cleanup_endpoint(endpoint_name: str = "heart-disease-endpoint"):
    """Delete the endpoint to stop billing."""
    ml_client = get_ml_client()

    print(f"\n🗑️ Deleting endpoint: {endpoint_name}")
    try:
        ml_client.online_endpoints.begin_delete(name=endpoint_name).result()
        print("   ✅ Endpoint deleted (no more charges)")
    except Exception as e:
        if "not found" in str(e).lower():
            print("   ℹ️ Endpoint doesn't exist (already deleted)")
        else:
            raise


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Azure ML Pipeline")
    parser.add_argument(
        "action",
        choices=["run", "deploy", "test", "cleanup"],
        help="What to do: run pipeline, deploy model, test endpoint, or cleanup",
    )

    args = parser.parse_args()

    if args.action == "run":
        run_pipeline()
    elif args.action == "deploy":
        deploy_model()
    elif args.action == "test":
        test_endpoint()
    elif args.action == "cleanup":
        cleanup_endpoint()
