## Building Network Security System with ETL Pipelines.

Purpose of the project:
To create a reproducible, production-minded Machine Learning pipeline that can process network telemetry stored in MongoDB, validate and transform it for training, handle class imbalance, produce stable model artifacts (preprocessor + model), track all experiments and artifacts with MLflow (S3 as artifact store), and expose a lightweight FastAPI endpoint to get predictions from uploaded CSVs. The aim is to demonstrate real-world MLOps and data engineering patterns for network security (anomaly/attack detection) use cases.

Objective (measurable & concrete)
1) Data ingestion from MongoDB into a reproducible feature store (raw CSVs: ingested/train.csv, ingested/test.csv).

2) Enforce schema & validate that train/test have expected columns + numeric features; emit report.yaml/report.json artifacts.

3) Data transformation: imputing, scaling (RobustScaler), encoding/mapping target value, handling class imbalance with SMOTETomek, and produce train.npy & test.npy arrays and preprocessing.pkl.

4) Model training & evaluation inside a pipeline (save model.pkl), log parameters/metrics/artifacts to MLflow and persist artifacts to S3.

5) Deployment: serve inference through app.py (FastAPI). Accept CSV uploads, produce prediction_output/output.csv and an HTML table response.

6) Reproducibility & observability: environment via .env + MLflow tracking server, and robust logging + custom exception handling via networkscurity.exception and networkscurity.logging.
