# Projet #3 — ETL + Airflow + Postgres (Docker) — Learn Notes

## Objectif portfolio

Montrer un pipeline orchestré “pro” : **extract → transform → validate → load**.

Attendus “portfolio” :

- orchestration via **Airflow** (DAGs, scheduling, retries, logs)
- stockage dans **Postgres** avec **schéma SQL**, **contraintes**, **indexes**
- **idempotence** (rejouer sans dupliquer)
- **data quality checks** (fail fast)
- séparation claire des zones : **raw** (brut) vs **curated** (analytique)

---

## Architecture (mental model)

### Services

- **Airflow webserver** : UI + API
- **Airflow scheduler** : planifie/exécute les tâches
- **Postgres metadata** : état Airflow (runs, tasks, logs, connexions…)
- **Postgres warehouse** : base cible pour les données du pipeline (raw + curated)

### Zones de données

- **Raw** : payloads JSON tels que récupérés de la source (reproductible, audit)
- **Curated** : tables “propres” (colonnes typées, prêtes pour analyse)

---

## [DE] ETL : extract → transform → load

### Extract

Objectif : **récupérer les données** depuis une source externe.

Bonnes pratiques :

- garder un “trace” : timestamp d’extraction, source, paramètres (ville, subreddit…)
- stocker le brut en JSON (raw) pour faciliter debug/replay
- gérer les erreurs réseau (timeout, retry)

### Transform

Objectif : **convertir** le raw en colonnes exploitables.

Exemples courants :

- parsing JSON → colonnes typées
- normalisation des dates (UTC)
- déduplication logique
- enrichissement (ex: extraire `temp`, `humidity`, etc.)

### Validate (data quality)

Objectif : éviter de charger des données incohérentes.

Checks typiques :

- **null checks** : colonnes obligatoires non nulles
- **range checks** : température plausible, score dans [0,1], etc.
- **duplicate checks** : clés uniques (ex: `(city, dt_iso)`)
- **freshness** : données pas trop anciennes

Principe : **fail fast**.

- si un check échoue → la task Airflow doit échouer
- on préfère “pas de données” plutôt que “données mauvaises”

### Load

Objectif : écrire en base warehouse.

Bonnes pratiques :

- transactions
- schéma SQL “contractuel” (types, contraintes)
- idempotence via clés uniques + UPSERT

---

## [DE] Idempotence (le concept clé)

### Définition

Un pipeline idempotent donne **le même résultat** quand on le relance plusieurs fois avec les mêmes entrées.

### Pourquoi c’est critique

- Airflow peut relancer (retry)
- on peut backfill / rerun un DAG
- un run partiel (extract ok, load échoue) ne doit pas casser l’état final

### Implémentation classique en Postgres

1) définir une **clé d’unicité** (unique constraint) sur la table target

- ex (weather_daily) : `(city, dt_iso)`
- ex (reddit_posts) : `post_id` (ou `(subreddit, post_id)` selon le besoin)

1) charger avec **UPSERT**

- `INSERT ... ON CONFLICT (...) DO UPDATE ...`

1) prouver l’idempotence

- relancer 2 fois le DAG
- comparer les counts avant/après : **pas d’augmentation due à des doublons**

---

## [SQL/Postgres] Schéma, contraintes, indexes

### Schéma = contrat

Le schéma définit :

- types (timestamp, numeric, text)
- colonnes obligatoires (`NOT NULL`)
- clés uniques (`UNIQUE`)

C’est la base de la robustesse : même si le code “bug”, la DB protège.

### Indexes

Un index accélère les requêtes et parfois les checks.

- index utile sur les colonnes filtrées souvent (`dt_iso`, `city`, `subreddit`)
- l’index de contrainte unique sert aussi à détecter rapidement les conflits (UPSERT)

---

## [Airflow] Concepts à maîtriser

### DAG

Un **DAG** est un graphe de tâches orienté (ordre d’exécution).

- un DAG = pipeline
- une task = unité de travail (extract / transform / check / load)

### Scheduling

- `schedule` = rythme d’exécution (cron / interval / manuel)
- `start_date` + `catchup` contrôlent le backfill

Idée simple pour portfolio :

- démarrage manuel
- ou un schedule quotidien

### Retries

- utile pour erreurs transitoires (réseau API)
- `retries=...`, `retry_delay=...`

Bon pattern :

- retry sur **extract** (réseau)
- moins sur **transform/load** (souvent déterministes)

### Logs

- chaque task écrit des logs consultables dans l’UI
- logs = outil #1 de debug

Règle : logguer les infos “observables” :

- paramètres (ville/subreddit)
- nombre de lignes traitées
- anomalies (nulls, duplicates)

### Connections / Variables

- **Connections** : credentials DB / API
- **Variables** : paramètres non sensibles (ex: city)

Bonnes pratiques :

- secrets via variables d’environnement / secrets manager (en local via `.env` / compose)
- ne pas committer de clés API

---

## [Docker Compose] Pourquoi et comment (Windows)

### Pourquoi Compose

- même environnement pour tout le monde
- versions et dépendances reproductibles
- lancement en 2 commandes (init + up)

### Séparer metadata vs warehouse

- metadata Airflow = “système Airflow”
- warehouse = “données business”

Avantages :

- éviter mélange des responsabilités
- possibilité d’écraser / reset warehouse sans casser Airflow

---

## [APIs] Clés et gestion

### OpenWeatherMap

- simple et rapide
- clé via variable d’environnement (ex: `OPENWEATHERMAP_API_KEY`)

### Reddit

- option “public JSON” possible sans auth (limites / restrictions)
- si auth : créer une app Reddit et utiliser client_id/secret

Règle :

- la clé n’est **jamais** hardcodée
- elle est injectée via env vars / Airflow connection

---

## Mini-exercice obligatoire — preuve d’idempotence

### But

Relancer le pipeline 2 fois et prouver : **0 duplicate**.

### Procédure type

1) run #1 : exécuter extract puis transform/load
1) vérifier en SQL : counts et clés uniques
1) run #2 : relancer les mêmes DAGs
1) vérifier : counts compatibles et aucune violation d’unicité

### Ce que tu dois pouvoir expliquer

- pourquoi des duplicates arrivent sans protection
- pourquoi `UNIQUE + UPSERT` résout le problème
- comment Airflow retries/backfill rendent l’idempotence indispensable

---

## Checklist “je sais expliquer”

- ETL : rôle de extract/transform/validate/load
- Raw vs Curated : pourquoi garder les payloads bruts
- Idempotence : unique keys + upsert + preuve
- Airflow : DAG, tasks, schedule, retries, logs
- Postgres : schéma, contraintes, indexes
- Docker Compose : reproductibilité + séparation metadata/warehouse
