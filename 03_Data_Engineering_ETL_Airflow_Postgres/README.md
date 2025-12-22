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

### Accéder à l’UI

- URL : <http://localhost:8080>
- Login : `admin`
- Password : `admin`

### Vérifier le DAG

- DAG : `hello_world`
- Lance-le manuellement (bouton Play) puis vérifie les logs de la task `hello`.
