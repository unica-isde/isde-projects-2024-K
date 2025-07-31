import json
from fastapi import FastAPI, Request, UploadFile, File, Query
from PIL import Image, ImageEnhance
import base64
import shutil
import uuid
import os
import io
import matplotlib.pyplot as plt
import numpy as np
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.config import Configuration
from app.forms.upload_form import UploadForm
from app.forms.classification_form import ClassificationForm
from app.ml.classification_utils import classify_image
from app.utils import list_images


app = FastAPI()
config = Configuration()
UPLOAD_FOLDER = os.path.join(Configuration().image_folder_path)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/info")
def info() -> dict[str, list[str]]:
    """Returns a dictionary with the list of models and
    the list of available image files."""
    list_of_images = list_images()
    list_of_models = Configuration.models
    data = {"models": list_of_models, "images": list_of_images}
    return data


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """The home page of the service."""
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/classifications")
def create_classify(request: Request):
    return templates.TemplateResponse(
        "classification_select.html",
        {"request": request, "images": list_images(), "models": Configuration.models},
    )


@app.post("/classifications")
async def request_classification(request: Request):
    form = ClassificationForm(request)
    await form.load_data()
    image_id = form.image_id
    model_id = form.model_id
    classification_scores = classify_image(model_id=model_id, img_id=image_id)

    # Unique result ID
    result_id = f"{uuid.uuid4().hex}_{image_id}_{model_id}"
    output_folder = "app/static/downloads"
    os.makedirs(output_folder, exist_ok=True)

    # Save JSON
    with open(os.path.join(output_folder, f"{result_id}.json"), "w") as f:
        json.dump(classification_scores, f)

    # Generate plot
    labels = [r[0] for r in classification_scores]
    scores = [r[1] for r in classification_scores]
    plt.figure()
    plt.barh(labels[::-1], scores[::-1])
    plt.title("Top-5 Predictions")
    plt.xlabel("Confidence")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.savefig(os.path.join(output_folder, f"{result_id}.png"))
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return templates.TemplateResponse(
        "classification_output.html",
        {
            "request": request,
            "image_id": image_id,
            "model_id": model_id,
            "classification_scores": json.dumps(classification_scores),
            "chart_data": image_base64,
            "result_id": result_id,
        },
    )

@app.get("/upload", response_class = HTMLResponse)
def upload_form(request : Request):
    return templates.TemplateResponse("upload_form.html",
                                      {"request": request, "images": list_images(), "models": Configuration.models})

@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(request: Request, file: UploadFile = File(...)):
    # Create unique name and path
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        # Save uploaded image
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        form_data = await request.form()
        model_id = form_data.get("model_id", "resnet18")

        # Run classification
        results = classify_image(model_id, filename)

        # Save results with unique ID
        result_id = filename
        output_folder = "app/static/downloads"
        os.makedirs(output_folder, exist_ok=True)

        # Save JSON
        with open(os.path.join(output_folder, f"{result_id}.json"), "w") as f:
            json.dump(results, f)

        # Save PNG
        labels = [r[0] for r in results]
        scores = [r[1] for r in results]
        plt.figure()
        plt.barh(labels[::-1], scores[::-1])
        plt.title("Top-5 Predictions")
        plt.xlabel("Confidence (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{result_id}.png"))
        plt.close()

        return templates.TemplateResponse(
            "upload_result.html",
            {
                "request": request,
                "image_url": f"/static/imagenet_subset/{filename}",
                "results": results,
                "result_id": result_id,
            },
        )

    except Exception as e:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

        # Optional: log the error on server for debugging
        print(f"Upload error: {e}")

        # Show error to user
        return templates.TemplateResponse(
            "upload_result.html",
            {
                "request": request,
                "error": "An error occurred while processing your image. Please try again with a valid file.",
            },
            status_code=400
        )

@app.get("/histogram", response_class=HTMLResponse)
def histogram_form(request: Request):
    from app.utils import list_images
    return templates.TemplateResponse(
        "histogram_form.html",
        {"request": request, "images": list_images()}
    )

@app.post("/histogram", response_class=HTMLResponse)
async def handle_histogram(request: Request):
    form = await request.form()
    image_id = form["image_id"]

    from app.ml.histogram_utils import generate_histogram

    histogram_path = generate_histogram(image_id)

    return templates.TemplateResponse(
        "histogram_result.html",
        {
            "request": request,
            "image_id": image_id,
            "image_path": f"/static/imagenet_subset/{image_id}",
            "histogram_path": f"/static/generated/{histogram_path}",
        },
    )

@app.get("/transform", response_class=HTMLResponse)
def transform_form(request: Request):
    return templates.TemplateResponse(
        "transform_form.html",
        {"request": request, "images": list_images()}
    )

@app.post("/transform", response_class=HTMLResponse)
async def handle_transform(request: Request):
    form = await request.form()
    image_id = form["image_id"]
    brightness = float(form.get("brightness", 1.0))
    contrast = float(form.get("contrast", 1.0))
    color = float(form.get("color", 1.0))
    sharpness = float(form.get("sharpness", 1.0))

    image_path = os.path.join(Configuration().image_folder_path, image_id)
    img = Image.open(image_path).convert("RGB")

    # Apply transformations
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(color)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)

    # Save with unique name
    unique_id = uuid.uuid4().hex
    transformed_name = f"transformed_{unique_id}_{image_id}"
    output_path = os.path.join("app/static/generated", transformed_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)

    return templates.TemplateResponse(
        "transform_result.html",
        {
            "request": request,
            "image_id": image_id,
            "original_path": f"/static/imagenet_subset/{image_id}",
            "transformed_path": f"/static/generated/{transformed_name}",
            "params": {
                "brightness": brightness,
                "contrast": contrast,
                "color": color,
                "sharpness": sharpness
            }
        },
    )

@app.get("/download/json/{result_id}")
def download_json(result_id: str):
    json_path = os.path.join("app/static/downloads", f"{result_id}.json")
    return FileResponse(json_path, media_type="application/json", filename="results.json")


@app.get("/download/plot/{result_id}")
def download_plot(result_id: str):
    png_path = os.path.join("app/static/downloads", f"{result_id}.png")
    return FileResponse(png_path, media_type="image/png", filename="results.png")

