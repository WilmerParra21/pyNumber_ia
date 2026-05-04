# pyNumber_ia

Pequeño proyecto educativo para clasificar dígitos manuscritos usando una red neuronal pre-entrenada tipo MNIST.

## Composición del proyecto

- `src/pages/index.astro`: página principal que carga el componente del clasificador.
- `src/components/MnistClassifier.astro`: interfaz de usuario, carga de archivos, validación de imagen y llamadas a la API.
- `api/index.py`: backend FastAPI que procesa la imagen, normaliza el dígito y devuelve la predicción.
- `public/trained_network.pkl`: modelo neuronal pre-entrenado usado por la API.
- `astro.config.mjs`: configuración de Astro y proxy en desarrollo.
- `vercel.json`: configuración de despliegue en Vercel, incluyendo function timeout y archivos incluidos.
- `requirements.txt`: dependencias de Python para la API.

## Formato de imágenes soportado

El proyecto acepta imágenes estándar (`PNG`, `JPG`, `WEBP`) y convierte automáticamente el tamaño al formato esperado por el modelo:

- Se redimensiona a `28 x 28` píxeles.
- Debe ser escala de grises.
- Se normaliza internamente como un vector de `784` valores entre `0.0` y `1.0`.

> En la UI se valida en vivo que el archivo sea una imagen válida en escala de grises. El tamaño se convierte automáticamente a 28x28 antes de enviarla al backend.

## Instalación y ejecución local

1. Clonar el repositorio:

```bash
git clone https://github.com/WilmerParra21/pyNumber_ia.git
cd pyNumber_ia
```

2. Instalar dependencias de Python:

```bash
python -m pip install -r requirements.txt
```

3. Ejecutar el backend FastAPI:

```bash
python -m uvicorn api.index:app --reload --port 8000
```

4. En otra terminal, instalar dependencias de Node.js y arrancar Astro:

```bash
npm install
npm run dev
```

5. Abrir la aplicación en el navegador:

```text
http://localhost:3000
```

## Uso

- Selecciona o arrastra una sola imagen.
- La UI valida que el archivo sea una imagen válida y esté en escala de grises.
- El tamaño se convierte automáticamente a 28x28 píxeles antes de enviarla al backend.
- Después de la conversión, el backend devuelve la predicción del dígito.

## Notas

- En producción, el backend se despliega como función de Vercel.
- El proyecto usa `FastAPI` para la API y `Astro` para el frontend.
