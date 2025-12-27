export type ProjectLink = {
  label: string;
  href: string;
};

export type ProjectMetric = {
  label: string;
  value: string;
};

export type Project = {
  slug: string;
  title: string;
  subtitle: string;
  year: string;
  stack: string[];
  highlights: string[];
  results?: string[];
  metrics: ProjectMetric[];
  links: ProjectLink[];
  demoUrl?: string;
  demoEmbedUrl?: string;
  readmePath?: string;
  readmeUrl?: string;
  coverImage?: {
    src: string;
    alt: string;
  };
};

export const projects: Project[] = [
  {
    slug: "ml-churn",
    title: "Projet #1 — ML Classique (Churn)",
    subtitle: "Pipeline de classification + évaluation (ROC-AUC, F1) + prévention du data leakage.",
    year: "2025",
    stack: ["Python", "pandas", "scikit-learn"],
    highlights: [
      "Pipeline de prétraitement + modèle",
      "Validation (Stratified K-Fold)",
      "Métriques orientées business (rappel, ROC-AUC)",
    ],
    metrics: [
      { label: "Objectif", value: "Réduire le churn" },
      { label: "Type", value: "Classification binaire" },
    ],
    results: [
      "Pipeline complet (prétraitement + modèle) prêt pour production",
      "Validation robuste (Stratified K-Fold) et métriques orientées business",
    ],
    links: [
      {
        label: "Code (GitHub)",
        href: "https://github.com/soboure69/data-science-portfolio/tree/main/01_ML_Classique_Churn",
      },
    ],
    readmePath: "../01_ML_Classique_Churn/README.md",
    readmeUrl:
      "https://raw.githubusercontent.com/soboure69/data-science-portfolio/master/01_ML_Classique_Churn/README.md",
    coverImage: {
      src: "/covers/ml-churn.svg",
      alt: "Aperçu du projet churn (ML classique)",
    },
  },
  {
    slug: "dl-nlp-sentiment",
    title: "Projet #2 — Deep Learning (NLP Sentiment)",
    subtitle: "Modèle BiLSTM pour classification de sentiment + production Streamlit.",
    year: "2025",
    stack: ["Python", "TensorFlow/Keras", "Streamlit"],
    highlights: [
      "Préparation texte + embeddings",
      "Régularisation (Dropout/EarlyStopping)",
      "Packaging artefacts (tokenizer + modèle)",
    ],
    metrics: [
      { label: "Objectif", value: "Analyse sentiment" },
      { label: "Type", value: "NLP (classification)" },
    ],
    results: [
      "Modèle entraîné + artefacts (tokenizer, modèle) packagés pour l’inférence",
      "App de démo pour tester le sentiment sur des textes",
    ],
    links: [
      {
        label: "Code (GitHub)",
        href: "https://github.com/soboure69/data-science-portfolio/tree/main/02_DL_NLP_Sentiment",
      },
    ],
    demoUrl: "https://data-science-portfolio-9trrfmqurfwu3p3hl7ibwz.streamlit.app/",
    demoEmbedUrl: "https://data-science-portfolio-9trrfmqurfwu3p3hl7ibwz.streamlit.app/",
    readmePath: "../02_DL_NLP_Sentiment/README.md",
    readmeUrl:
      "https://raw.githubusercontent.com/soboure69/data-science-portfolio/master/02_DL_NLP_Sentiment/README.md",
    coverImage: {
      src: "/covers/dl-nlp-sentiment.svg",
      alt: "Aperçu du projet NLP sentiment (Deep Learning)",
    },
  },
  {
    slug: "etl-airflow-postgres",
    title: "Projet #3 — Data Engineering (ETL Airflow + Postgres)",
    subtitle: "Pipelines ETL idempotents (OpenWeatherMap + Reddit) orchestrés par Airflow.",
    year: "2025",
    stack: ["Python", "Airflow", "PostgreSQL", "Docker"],
    highlights: [
      "Orchestration DAGs",
      "Schéma warehouse + tables raw/analytics",
      "Contrôles qualité + idempotence (0 duplicats)",
    ],
    metrics: [
      { label: "Objectif", value: "Warehouse analytique" },
      { label: "Pattern", value: "ELT/ETL + idempotence" },
    ],
    results: [
      "Schéma warehouse (raw + analytics) + contrôles qualité",
      "Chargement idempotent (zéro duplicat) orchestré par DAGs",
    ],
    links: [
      {
        label: "Code (GitHub)",
        href: "https://github.com/soboure69/data-science-portfolio/tree/main/03_Data_Engineering_ETL_Airflow_Postgres",
      },
    ],
    readmePath: "../03_Data_Engineering_ETL_Airflow_Postgres/README.md",
    readmeUrl:
      "https://raw.githubusercontent.com/soboure69/data-science-portfolio/master/03_Data_Engineering_ETL_Airflow_Postgres/README.md",
    coverImage: {
      src: "/covers/etl-airflow-postgres.svg",
      alt: "Aperçu du projet ETL (Airflow + Postgres)",
    },
  },
  {
    slug: "dashboard-recommendation",
    title: "Projet #4 — Dashboard Recommandation (Dash)",
    subtitle: "Recommandation content-based (TF-IDF + cosine similarity) + dashboard interactif.",
    year: "2025",
    stack: ["Python", "Dash", "Plotly", "scikit-learn", "Render"],
    highlights: [
      "Moteur de reco content-based",
      "UX: filtres + gestion cas 0 reco",
      "Déploiement cloud (Render)",
    ],
    metrics: [
      { label: "Démo", value: "Live" },
      { label: "Approche", value: "TF-IDF + cosine" },
    ],
    results: [
      "Moteur de recommandation content-based (TF-IDF + cosine similarity)",
      "Dashboard interactif (filtres + gestion cas limites) déployé sur Render",
    ],
    links: [
      { label: "Démo (Render)", href: "https://dashboard-recommendation-a1ds.onrender.com" },
      {
        label: "Code (GitHub)",
        href: "https://github.com/soboure69/data-science-portfolio/tree/main/04_Dashboard_Recommendation",
      },
    ],
    demoUrl: "https://dashboard-recommendation-a1ds.onrender.com",
    demoEmbedUrl: "https://dashboard-recommendation-a1ds.onrender.com",
    readmePath: "../04_Dashboard_Recommendation/README.md",
    readmeUrl:
      "https://raw.githubusercontent.com/soboure69/data-science-portfolio/master/04_Dashboard_Recommendation/README.md",
    coverImage: {
      src: "/covers/dashboard-recommendation.svg",
      alt: "Aperçu du dashboard de recommandation",
    },
  },
];

export function getProjectBySlug(slug: string): Project | undefined {
  return projects.find((p) => p.slug === slug);
}
