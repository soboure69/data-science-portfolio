# Projet #3 — Data Engineering : ETL + Airflow + Postgres (Docker)

## J1 — Démarrage (Airflow UI + DAG minimal)

### Prérequis

- Docker Desktop installé et lancé

### Lancer Airflow

Depuis ce dossier :

```bash
docker compose up airflow-init
docker compose up
```

### Postgres (warehouse)

Un second Postgres est lancé pour stocker les données ETL :

- Service docker : `warehouse`
- DB : `warehouse`
- User : `warehouse`
- Password : `warehouse`

Une table `weather_raw` est créée automatiquement au démarrage via `sql/`.

### Airflow Connection

La connection Airflow `warehouse_postgres` est créée automatiquement au `airflow-init`.
Tu peux la voir dans l’UI : Admin → Connections.

### Accéder à l’UI

- URL : <http://localhost:8080>
- Login : `admin`
- Password : `admin`

### Vérifier le DAG

- DAG : `hello_world`
- Lance-le manuellement (bouton Play) puis vérifie les logs de la task `hello`.

## J3 — Extract Weather API (OpenWeatherMap)

### 1) Créer une clé API OpenWeatherMap

- Créer un compte : <https://home.openweathermap.org/users/sign_up>
- Récupérer la clé : <https://home.openweathermap.org/api_keys>

Note : la clé peut prendre quelques minutes à être activée.

### 2) Configurer les Airflow Variables

Dans l’UI Airflow : Admin → Variables

- `OPENWEATHER_API_KEY` : ta clé
- `OPENWEATHER_CITY` : `Lyon`
- `OPENWEATHER_COUNTRY` : `FR`

### 3) Lancer le DAG

- DAG : `extract_openweathermap_weather`
- Lance-le manuellement
- Vérifie ensuite que des lignes sont insérées dans `warehouse.public.weather_raw`

## J4 — Transform + checks qualité + load

### Objectif

- Transformer les JSON bruts (`weather_raw`) en colonnes analytiques (`weather_daily`)
- Appliquer des checks qualité (nulls / ranges / duplicates)
- Charger de façon idempotente (pas de duplicats sur `(city, country, obs_date)`)

### Lancer le DAG

- DAG : `transform_load_weather_daily`
- Lance-le manuellement après avoir exécuté l’extractor

### Validation attendue

- Task `create_weather_daily_table` : Success
- Task `transform_and_load_weather_daily` : Success
- Table `warehouse.public.weather_daily` contient des lignes
