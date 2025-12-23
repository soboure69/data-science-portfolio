from __future__ import annotations

import json
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook


def _extract_weather() -> None:
    api_key = Variable.get("OPENWEATHER_API_KEY")
    city = Variable.get("OPENWEATHER_CITY", default_var="Lyon")
    country = Variable.get("OPENWEATHER_COUNTRY", default_var="FR")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": f"{city},{country}",
        "appid": api_key,
        "units": "metric",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    hook = PostgresHook(postgres_conn_id="warehouse_postgres")
    sql = (
        "INSERT INTO public.weather_raw (source, city, country, retrieved_at, payload) "
        "VALUES (%s, %s, %s, now(), %s::jsonb)"
    )

    hook.run(sql, parameters=("openweathermap", city, country, json.dumps(payload)))


with DAG(
    dag_id="extract_openweathermap_weather",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["j3", "extract", "weather"],
) as dag:
    extract_weather = PythonOperator(task_id="extract_weather", python_callable=_extract_weather)
