import mlflow

mlflow.set_tracking_uri("http://localhost:5050") 
mlflow.set_experiment("Infra_Test")

with mlflow.start_run(run_name="Connection_Check"):
    mlflow.log_param("role", "Architect_A")
    mlflow.log_metric("status", 1.0)
    print(">>> MLflow connection test completed successfully!")