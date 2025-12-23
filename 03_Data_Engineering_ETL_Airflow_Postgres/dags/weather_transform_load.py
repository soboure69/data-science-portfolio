from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook


def _create_weather_daily_table() -> None:
    hook = PostgresHook(postgres_conn_id="warehouse_postgres")
    hook.run(
        """
        CREATE TABLE IF NOT EXISTS public.weather_daily (
          id BIGSERIAL PRIMARY KEY,
          source VARCHAR(50) NOT NULL,
          city VARCHAR(120) NOT NULL,
          country VARCHAR(2),
          obs_date DATE NOT NULL,
          obs_ts TIMESTAMPTZ,
          temp_c DOUBLE PRECISION,
          humidity INTEGER,
          pressure INTEGER,
          wind_speed DOUBLE PRECISION,
          weather_main VARCHAR(50),
          weather_description VARCHAR(200),
          retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          payload JSONB
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_weather_daily_city_country_date
          ON public.weather_daily (city, country, obs_date);
        """
    )


def _transform_and_load() -> None:
    hook = PostgresHook(postgres_conn_id="warehouse_postgres")

    rows = hook.get_records(
        """
        SELECT city, country, retrieved_at, payload
        FROM public.weather_raw
        WHERE source = 'openweathermap'
        ORDER BY retrieved_at DESC
        LIMIT 50
        """
    )

    if not rows:
        raise ValueError("No rows found in public.weather_raw. Run the extractor DAG first.")

    batch = []
    for city, country, retrieved_at, payload in rows:
        if payload is None:
            continue

        if isinstance(payload, str):
            data = json.loads(payload)
        else:
            data = payload

        dt_unix = data.get("dt")
        if dt_unix is None:
            continue

        obs_ts = datetime.fromtimestamp(float(dt_unix), tz=timezone.utc)
        obs_date = obs_ts.date()

        main = data.get("main") or {}
        wind = data.get("wind") or {}
        weather0 = (data.get("weather") or [{}])[0] or {}

        temp_c = main.get("temp")
        humidity = main.get("humidity")
        pressure = main.get("pressure")
        wind_speed = wind.get("speed")
        weather_main = weather0.get("main")
        weather_description = weather0.get("description")

        # Basic quality checks (fail fast)
        if city is None or str(city).strip() == "":
            continue
        if country is not None and len(str(country)) != 2:
            raise ValueError(f"Invalid country code: {country}")
        if temp_c is None:
            raise ValueError("Missing main.temp in OpenWeather payload")
        if humidity is not None and not (0 <= int(humidity) <= 100):
            raise ValueError(f"Invalid humidity: {humidity}")

        batch.append(
            (
                "openweathermap",
                city,
                country,
                obs_date,
                obs_ts.isoformat(),
                float(temp_c),
                int(humidity) if humidity is not None else None,
                int(pressure) if pressure is not None else None,
                float(wind_speed) if wind_speed is not None else None,
                weather_main,
                weather_description,
                json.dumps(data),
            )
        )

    if not batch:
        raise ValueError("No valid rows to load after transformation.")

    # Idempotent load: upsert by (city, country, obs_date)
    hook.insert_rows(
        table="public.weather_daily",
        rows=batch,
        target_fields=[
            "source",
            "city",
            "country",
            "obs_date",
            "obs_ts",
            "temp_c",
            "humidity",
            "pressure",
            "wind_speed",
            "weather_main",
            "weather_description",
            "payload",
        ],
        commit_every=100,
        replace=False,
        replace_index=["city", "country", "obs_date"],
    )

    # Post-load checks
    dup_count = hook.get_first(
        """
        SELECT COUNT(*)
        FROM (
          SELECT city, country, obs_date, COUNT(*) AS n
          FROM public.weather_daily
          GROUP BY city, country, obs_date
          HAVING COUNT(*) > 1
        ) t
        """
    )[0]
    if dup_count and int(dup_count) > 0:
        raise ValueError(f"Duplicate keys detected in weather_daily: {dup_count}")


with DAG(
    dag_id="transform_load_weather_daily",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["j4", "transform", "quality", "load"],
) as dag:
    create_table = PythonOperator(
        task_id="create_weather_daily_table",
        python_callable=_create_weather_daily_table,
    )

    transform_load = PythonOperator(
        task_id="transform_and_load_weather_daily",
        python_callable=_transform_and_load,
    )

    create_table >> transform_load
