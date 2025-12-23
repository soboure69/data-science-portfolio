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
