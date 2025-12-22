from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def _hello() -> None:
    print("Hello from Airflow!")


with DAG(
    dag_id="hello_world",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["j1", "intro"],
) as dag:
    hello = PythonOperator(task_id="hello", python_callable=_hello)
