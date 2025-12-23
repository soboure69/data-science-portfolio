from __future__ import annotations

import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook


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


def _transform_and_load_reddit_posts() -> None:
    hook = PostgresHook(postgres_conn_id="warehouse_postgres")
    _ensure_tables(hook)

    rows = hook.get_records(
        """
        SELECT subreddit, retrieved_at, payload
        FROM public.reddit_raw
        ORDER BY retrieved_at DESC
        LIMIT 20
        """
    )

    if not rows:
        raise ValueError("No rows found in public.reddit_raw. Run extract_reddit_raw first.")

    batch: list[tuple] = []

    for subreddit, retrieved_at, payload in rows:
        data = json.loads(payload) if isinstance(payload, str) else payload
        children = (((data or {}).get("data") or {}).get("children") or [])

        for item in children:
            post = (item or {}).get("data") or {}
            post_id = post.get("id")
            if not post_id:
                continue

            title = post.get("title")
            author = post.get("author")
            score = post.get("score")
            num_comments = post.get("num_comments")
            created_utc = post.get("created_utc")
            permalink = post.get("permalink")
            url = post.get("url")

            if score is not None:
                score = int(score)
            if num_comments is not None:
                num_comments = int(num_comments)

            created_ts = None
            if created_utc is not None:
                created_ts = datetime.utcfromtimestamp(float(created_utc)).isoformat() + "+00:00"

            batch.append(
                (
                    subreddit,
                    str(post_id),
                    title,
                    author,
                    score,
                    num_comments,
                    created_ts,
                    f"https://www.reddit.com{permalink}" if permalink else None,
                    url,
                    json.dumps(post),
                )
            )

    if not batch:
        raise ValueError("No posts parsed from reddit_raw payloads.")

    upsert_sql = """
    INSERT INTO public.reddit_posts (
      subreddit,
      post_id,
      title,
      author,
      score,
      num_comments,
      created_utc,
      permalink,
      url,
      payload
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s::timestamptz, %s, %s, %s::jsonb)
    ON CONFLICT (subreddit, post_id)
    DO UPDATE SET
      title = EXCLUDED.title,
      author = EXCLUDED.author,
      score = EXCLUDED.score,
      num_comments = EXCLUDED.num_comments,
      created_utc = EXCLUDED.created_utc,
      permalink = EXCLUDED.permalink,
      url = EXCLUDED.url,
      payload = EXCLUDED.payload,
      retrieved_at = now();
    """

    conn = hook.get_conn()
    try:
        with conn.cursor() as cur:
            cur.executemany(upsert_sql, batch)
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Post-load duplicate check
    dup_count = hook.get_first(
        """
        SELECT COUNT(*)
        FROM (
          SELECT subreddit, post_id, COUNT(*) AS n
          FROM public.reddit_posts
          GROUP BY subreddit, post_id
          HAVING COUNT(*) > 1
        ) t
        """
    )[0]
    if dup_count and int(dup_count) > 0:
        raise ValueError(f"Duplicate keys detected in reddit_posts: {dup_count}")


with DAG(
    dag_id="transform_load_reddit_posts",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["j5", "transform", "load", "reddit"],
) as dag:
    transform_load = PythonOperator(
        task_id="transform_load_reddit_posts",
        python_callable=_transform_and_load_reddit_posts,
    )
