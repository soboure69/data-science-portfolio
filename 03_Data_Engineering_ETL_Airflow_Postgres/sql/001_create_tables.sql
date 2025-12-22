CREATE TABLE IF NOT EXISTS public.weather_raw (
  id BIGSERIAL PRIMARY KEY,
  source VARCHAR(50) NOT NULL DEFAULT 'openweathermap',
  city VARCHAR(120) NOT NULL,
  country VARCHAR(2),
  retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  payload JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_weather_raw_city_retrieved_at
  ON public.weather_raw (city, retrieved_at DESC);
