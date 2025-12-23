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
