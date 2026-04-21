from __future__ import annotations

import base64
import io
import os
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError


MNIST_CANVAS_SIZE = 28
MNIST_DEFAULT_DIGIT_SIZE = 20


class Network:
    """Red neuronal simple compatible con modelos MNIST serializados con pickle."""

    def __init__(self, sizes: list[int]) -> None:
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """Ejecuta inferencia para una entrada MNIST de forma 784x1."""
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a

    def SGD(
        self,
        training_data: list[tuple[np.ndarray, np.ndarray]],
        epochs: int,
        mini_batch_size: int,
        eta: float,
        test_data: list[tuple[np.ndarray, int]] | None = None,
    ) -> None:
        """Metodo incluido solo para compatibilidad con pickles de la clase original."""
        raise NotImplementedError("El entrenamiento no esta habilitado en la API.")

    def update_mini_batch(
        self, mini_batch: list[tuple[np.ndarray, np.ndarray]], eta: float
    ) -> None:
        """Metodo incluido solo para compatibilidad con pickles de la clase original."""
        raise NotImplementedError("El entrenamiento no esta habilitado en la API.")

    def backprop(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Metodo incluido solo para compatibilidad con pickles de la clase original."""
        raise NotImplementedError("El entrenamiento no esta habilitado en la API.")

    def evaluate(self, test_data: list[tuple[np.ndarray, int]]) -> int:
        """Evalua datos de prueba cuando se usa la clase fuera del endpoint."""
        test_results = [(int(np.argmax(self.feedforward(x))), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Funcion sigmoide usada por la red original."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    """Derivada de la sigmoide, conservada por compatibilidad."""
    return sigmoid(z) * (1 - sigmoid(z))


class NetworkUnpickler(pickle.Unpickler):
    """Carga modelos guardados desde notebooks donde Network vivia en __main__."""

    def find_class(self, module: str, name: str) -> Any:
        if name == "Network":
            return Network
        return super().find_class(module, name)


@dataclass(frozen=True)
class CandidatePrediction:
    """Resultado interno para comparar variantes de preprocesamiento."""

    image: Image.Image
    variant: str
    probabilities: list[float]
    prediction: int
    confidence: float
    quality: float


app = FastAPI(
    title="MNIST IA API",
    description="API para clasificar digitos manuscritos con un modelo pre-entrenado.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "public" / "trained_network.pkl"
MODEL: Network | None = None


def load_model() -> Network:
    """Carga el modelo una sola vez y lo reutiliza entre peticiones."""
    global MODEL

    if MODEL is not None:
        return MODEL

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No se encontro el modelo en {MODEL_PATH}. Copia trained_network.pkl en public/."
        )

    with MODEL_PATH.open("rb") as model_file:
        loaded_model = NetworkUnpickler(model_file).load()

    if not hasattr(loaded_model, "feedforward"):
        raise RuntimeError("El archivo .pkl no contiene una red compatible.")

    MODEL = loaded_model
    return MODEL


def image_to_grayscale(image: Image.Image) -> Image.Image:
    """Convierte cualquier imagen a gris, componiendo transparencias sobre blanco."""
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("L")

    return image.convert("L")


def resized_shape(source: np.ndarray, digit_size: int) -> tuple[int, int]:
    """Calcula el tamano manteniendo aspecto para encajar el digito en MNIST."""
    height, width = source.shape

    if height <= 0 or width <= 0:
        return (digit_size, digit_size)

    scale = digit_size / max(width, height)
    return (max(1, int(round(width * scale))), max(1, int(round(height * scale))))


def center_digit(canvas: np.ndarray) -> np.ndarray:
    """Centra el digito usando centro de masa, sin envolver pixeles."""
    total = float(np.sum(canvas))

    if total <= 0:
        return canvas

    y_indices, x_indices = np.indices(canvas.shape)
    center_y = float(np.sum(y_indices * canvas) / total)
    center_x = float(np.sum(x_indices * canvas) / total)
    target_center = (MNIST_CANVAS_SIZE - 1) / 2
    shift_y = int(round(target_center - center_y))
    shift_x = int(round(target_center - center_x))

    shifted = np.zeros_like(canvas)
    source_y_start = max(0, -shift_y)
    source_y_end = min(MNIST_CANVAS_SIZE, MNIST_CANVAS_SIZE - shift_y)
    source_x_start = max(0, -shift_x)
    source_x_end = min(MNIST_CANVAS_SIZE, MNIST_CANVAS_SIZE - shift_x)
    target_y_start = max(0, shift_y)
    target_y_end = min(MNIST_CANVAS_SIZE, MNIST_CANVAS_SIZE + shift_y)
    target_x_start = max(0, shift_x)
    target_x_end = min(MNIST_CANVAS_SIZE, MNIST_CANVAS_SIZE + shift_x)

    shifted[target_y_start:target_y_end, target_x_start:target_x_end] = canvas[
        source_y_start:source_y_end, source_x_start:source_x_end
    ]
    return shifted


def foreground_from_grayscale(grayscale: Image.Image, invert: bool) -> np.ndarray:
    """Devuelve una matriz donde el trazo es claro y el fondo oscuro."""
    contrasted = ImageOps.autocontrast(grayscale)
    source = np.asarray(contrasted, dtype=np.uint8)
    return 255 - source if invert else source


def should_invert_background(grayscale: Image.Image) -> bool:
    """Detecta si el fondo probable es claro usando las esquinas de la imagen."""
    source = np.asarray(grayscale, dtype=np.uint8)
    corner_size = max(1, min(source.shape) // 10)
    corners = np.concatenate(
        [
            source[:corner_size, :corner_size].ravel(),
            source[:corner_size, -corner_size:].ravel(),
            source[-corner_size:, :corner_size].ravel(),
            source[-corner_size:, -corner_size:].ravel(),
        ]
    )
    return float(np.median(corners)) > 127


def remove_border_components(mask: np.ndarray) -> np.ndarray:
    """Elimina componentes conectados que tocan el borde de la imagen."""
    cleaned = mask.astype(bool).copy()
    height, width = cleaned.shape
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        if cleaned[0, x]:
            queue.append((0, x))
        if cleaned[height - 1, x]:
            queue.append((height - 1, x))

    for y in range(height):
        if cleaned[y, 0]:
            queue.append((y, 0))
        if cleaned[y, width - 1]:
            queue.append((y, width - 1))

    while queue:
        y, x = queue.popleft()

        if not cleaned[y, x]:
            continue

        cleaned[y, x] = False

        for next_y, next_x in (
            (y - 1, x),
            (y + 1, x),
            (y, x - 1),
            (y, x + 1),
        ):
            if 0 <= next_y < height and 0 <= next_x < width and cleaned[next_y, next_x]:
                queue.append((next_y, next_x))

    return cleaned


def mask_quality(mask: np.ndarray) -> float:
    """Puntua si una mascara tiene proporcion y caja parecidas a un digito."""
    if not np.any(mask):
        return 0.0

    image_area = float(mask.size)
    area_ratio = float(np.sum(mask) / image_area)
    y_positions, x_positions = np.where(mask)
    bbox_height = int(y_positions.max()) - int(y_positions.min()) + 1
    bbox_width = int(x_positions.max()) - int(x_positions.min()) + 1
    bbox_area_ratio = float((bbox_height * bbox_width) / image_area)
    aspect = bbox_width / max(1, bbox_height)

    area_score = max(0.0, 1.0 - abs(area_ratio - 0.08) / 0.18)
    bbox_score = max(0.0, 1.0 - abs(bbox_area_ratio - 0.18) / 0.35)
    aspect_score = max(0.0, 1.0 - abs(aspect - 0.75) / 1.25)
    return (area_score * 0.45) + (bbox_score * 0.35) + (aspect_score * 0.20)


def extract_stroke_source(grayscale: Image.Image, invert: bool | None) -> tuple[np.ndarray, np.ndarray, str]:
    """Obtiene matriz de trazo claro y mascara del digito real."""
    source_gray = np.asarray(ImageOps.autocontrast(grayscale), dtype=np.uint8)
    polarities = [invert] if invert is not None else [True, False]
    candidates: list[tuple[float, np.ndarray, np.ndarray, str]] = []

    for use_invert in polarities:
        source = 255 - source_gray if use_invert else source_gray
        source = np.asarray(ImageOps.autocontrast(Image.fromarray(source)), dtype=np.uint8)
        threshold = max(18, int(float(source.max()) * 0.14))
        mask = source > threshold
        cleaned_mask = remove_border_components(mask)

        if not np.any(cleaned_mask):
            cleaned_mask = mask

        polarity = "dark-stroke" if use_invert else "light-stroke"
        candidates.append((mask_quality(cleaned_mask), source, cleaned_mask, polarity))

    _, best_source, best_mask, best_polarity = max(candidates, key=lambda item: item[0])
    return best_source, best_mask, best_polarity


def normalize_digit_image(
    content: bytes,
    *,
    digit_size: int = MNIST_DEFAULT_DIGIT_SIZE,
    invert: bool | None = None,
    center: bool = True,
) -> Image.Image:
    """Convierte una imagen al formato visual MNIST: fondo negro y trazo claro."""
    try:
        with Image.open(io.BytesIO(content)) as image:
            grayscale = image_to_grayscale(image)
            source, mask, _ = extract_stroke_source(grayscale, invert)

            if np.any(mask):
                y_positions, x_positions = np.where(mask)
                top = int(y_positions.min())
                bottom = int(y_positions.max()) + 1
                left = int(x_positions.min())
                right = int(x_positions.max()) + 1
                source = source[top:bottom, left:right]

            digit = Image.fromarray(source).resize(
                resized_shape(source, digit_size), Image.Resampling.LANCZOS
            )
            canvas_image = Image.new("L", (MNIST_CANVAS_SIZE, MNIST_CANVAS_SIZE), 0)
            offset = (
                (MNIST_CANVAS_SIZE - digit.width) // 2,
                (MNIST_CANVAS_SIZE - digit.height) // 2,
            )
            canvas_image.paste(digit, offset)

            canvas = np.asarray(canvas_image, dtype=np.float32)
            if center:
                canvas = center_digit(canvas)

            canvas = np.clip(canvas, 0, 255).astype(np.uint8)
            return Image.fromarray(canvas, mode="L")
    except UnidentifiedImageError as exc:
        raise ValueError("El archivo no es una imagen valida.") from exc


def image_to_vector(image: Image.Image) -> np.ndarray:
    """Convierte una imagen MNIST 28x28 a vector normalizado 784x1."""
    return (np.asarray(image, dtype=np.float32) / 255.0).reshape(784, 1)


def preprocess_image(content: bytes) -> np.ndarray:
    """Preprocesamiento principal conservado para uso externo o pruebas."""
    return image_to_vector(normalize_digit_image(content))


def to_probabilities(raw_output: np.ndarray) -> list[float]:
    """Transforma activaciones de salida en una distribucion de probabilidad estable."""
    values = np.asarray(raw_output, dtype=np.float64).reshape(-1)

    if values.size != 10:
        raise ValueError("El modelo debe devolver 10 valores, uno por digito.")

    non_negative = np.clip(values, 0.0, None)
    total = float(np.sum(non_negative))

    if total > 0:
        probabilities = non_negative / total
    else:
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        probabilities = exp_values / np.sum(exp_values)

    return [round(float(value), 6) for value in probabilities]


def mnist_quality(image: Image.Image) -> float:
    """Puntua si la imagen procesada se parece a MNIST.

    Penaliza fondos claros y parches rectangulares grandes; esos casos pueden
    provocar predicciones muy confiadas pero equivocadas.
    """
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    ink_mask = pixels > 0.12
    ink_ratio = float(np.mean(ink_mask))
    background_mean = float(
        np.mean(
            np.concatenate(
                [pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1]]
            )
        )
    )

    if not np.any(ink_mask):
        return 0.0

    y_positions, x_positions = np.where(ink_mask)
    bbox_area = float(
        (int(y_positions.max()) - int(y_positions.min()) + 1)
        * (int(x_positions.max()) - int(x_positions.min()) + 1)
    )
    fill_ratio = float(np.sum(ink_mask) / bbox_area)

    ink_score = max(0.0, 1.0 - abs(ink_ratio - 0.18) / 0.18)
    background_score = max(0.0, 1.0 - background_mean * 4.0)
    fill_score = max(0.0, 1.0 - max(0.0, fill_ratio - 0.42) / 0.35)
    return (ink_score * 0.45) + (background_score * 0.35) + (fill_score * 0.20)


def predict_candidates(model: Network, content: bytes) -> CandidatePrediction:
    """Evalua variantes y descarta las que no tienen forma visual MNIST."""
    candidates: list[CandidatePrediction] = []

    for digit_size in (18, 20, 22):
        for center in (True, False):
            for invert in (None, True, False):
                processed_image = normalize_digit_image(
                    content, digit_size=digit_size, center=center, invert=invert
                )
                quality = mnist_quality(processed_image)
                probabilities = to_probabilities(model.feedforward(image_to_vector(processed_image)))
                prediction = int(np.argmax(probabilities))
                confidence = probabilities[prediction]
                invert_label = "auto" if invert is None else str(invert).lower()
                candidates.append(
                    CandidatePrediction(
                        image=processed_image,
                        variant=(
                            f"size={digit_size};center={center};"
                            f"invert={invert_label};quality={quality:.2f}"
                        ),
                        probabilities=probabilities,
                        prediction=prediction,
                        confidence=confidence,
                        quality=quality,
                    )
                )

    reliable_candidates = [
        candidate for candidate in candidates if candidate.quality >= 0.70
    ]

    if reliable_candidates:
        return max(reliable_candidates, key=lambda candidate: candidate.confidence)

    return max(candidates, key=lambda candidate: (candidate.quality, candidate.confidence))


def image_to_data_url(image: Image.Image) -> str:
    """Serializa la imagen procesada para inspeccionarla desde el frontend."""
    buffer = io.BytesIO()
    image.resize((112, 112), Image.Resampling.NEAREST).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.get("/api/health")
def health() -> dict[str, str]:
    """Endpoint simple para verificar que la API esta activa."""
    return {"status": "ok"}


@app.post("/api/predict")
@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Recibe una o varias imagenes y devuelve la prediccion MNIST de cada una."""
    if not files:
        raise HTTPException(status_code=400, detail="Debes enviar al menos una imagen.")

    try:
        model = load_model()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results: list[dict[str, Any]] = []

    for uploaded_file in files:
        if uploaded_file.content_type and not uploaded_file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"{uploaded_file.filename} no parece ser una imagen.",
            )

        try:
            content = await uploaded_file.read()
            best = predict_candidates(model, content)
            results.append(
                {
                    "filename": uploaded_file.filename,
                    "prediction": best.prediction,
                    "confidence": best.confidence,
                    "probabilities": best.probabilities,
                    "processedImage": image_to_data_url(best.image),
                    "preprocessing": best.variant,
                }
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"No se pudo procesar {uploaded_file.filename}: {exc}",
            ) from exc

    response: dict[str, Any] = {"results": results}

    if len(results) == 1:
        response.update(results[0])

    return response
