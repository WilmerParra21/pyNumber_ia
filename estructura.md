
Estructura

/mnist-ia-web
├── api/                   # Backend (Python + FastAPI)
│   ├── index.py           # Endpoint principal y carga del modelo
│   └── requirements.txt   # Dependencias congeladas
├── public/                # Archivos estáticos y el Modelo
│   └── trained_network.pkl # TU MODELO (Cópialo aquí)
├── src/                   # Frontend (Astro + Vue/React/Svelte)
│   ├── components/        # UI: UploadZone, ResultCard, Spinner
│   ├── layouts/           # Layout minimalista
│   └── pages/             # index.astro (Página principal)
├── package.json           # Configuración de Astro y scripts
├── tailwind.config.mjs    # Estilo profesional (Earth tones / High-tech)
└── README.md              # Documentación del proyecto
