import json
from fastapi import FastAPI, Request, UploadFile, File
import shutil
import uuid
import os
import matplotlib.pyplot as plt
import numpy as np
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.config import Configuration
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
    return templates.TemplateResponse(
        "classification_output.html",
        {
            "request": request,
            "image_id": image_id,
            "classification_scores": json.dumps(classification_scores),
        },
    )

@app.get("/upload", response_class = HTMLResponse)
def upload_form(request : Request):
    return templates.TemplateResponse("upload_form.html",{"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(request: Request, file: UploadFile = File(...)):
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Use classify_image function
    from app.ml.classification_utils import classify_image
    results = classify_image("resnet18", filename) 

    return templates.TemplateResponse(
        "upload_result.html",
        {
            "request": request,
            "image_url": f"/static/imagenet_subset/{filename}",
            "results": results,
        },
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