from __future__ import annotations

import json
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook


def _parse_subreddits(raw: str) -> list[str]:
    subs = [s.strip() for s in raw.split(",")]
    return [s for s in subs if s]


def _ensure_tables(hook: PostgresHook) -> None:
    hook.run(
        """
        CREATE TABLE IF NOT EXISTS public.reddit_raw (
          id BIGSERIAL PRIMARY KEY,
          subreddit VARCHAR(100) NOT NULL,
          retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          payload JSONB NOT NULL
        );

        CREATE INDEX IF NOT EXISTS ix_reddit_raw_subreddit_retrieved_at
          ON public.reddit_raw (subreddit, retrieved_at DESC);

        CREATE TABLE IF NOT EXISTS public.reddit_posts (
          id BIGSERIAL PRIMARY KEY,
          subreddit VARCHAR(100) NOT NULL,
          post_id VARCHAR(20) NOT NULL,
          title TEXT,
          author VARCHAR(200),
          score INTEGER,
          num_comments INTEGER,
          created_utc TIMESTAMPTZ,
          permalink TEXT,
          url TEXT,
          retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          payload JSONB
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_reddit_posts_subreddit_post_id
          ON public.reddit_posts (subreddit, post_id);
        """
    )


def _extract_reddit_raw() -> None:
    subreddits = _parse_subreddits(
        Variable.get("REDDIT_SUBREDDITS", default_var="datascience,machinelearning,france")
    )
    limit = int(Variable.get("REDDIT_LIMIT", default_var="25"))
    user_agent = Variable.get(
        "REDDIT_USER_AGENT", default_var="data-science-portfolio-etl/0.1 (by u/your_username)"
    )

    hook = PostgresHook(postgres_conn_id="warehouse_postgres")
    _ensure_tables(hook)

    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }

    for subreddit in subreddits:
        url = f"https://www.reddit.com/r/{subreddit}/new.json"
        params = {"limit": limit}

        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code in (429, 403):
            raise ValueError(
                f"Reddit request blocked (status={resp.status_code}). "
                "Set a proper REDDIT_USER_AGENT Airflow Variable and retry."
            )
        resp.raise_for_status()

        payload = resp.json()

        hook.run(
            "INSERT INTO public.reddit_raw (subreddit, retrieved_at, payload) VALUES (%s, now(), %s::jsonb)",
            parameters=(subreddit, json.dumps(payload)),
        )


with DAG(
    dag_id="extract_reddit_raw",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["j5", "extract", "reddit"],
) as dag:
    extract = PythonOperator(task_id="extract_reddit_raw", python_callable=_extract_reddit_raw)
