import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.api.routes import predict, explain, simulate, cohort, health
from src.api.schemas import StudentInputSchema, PredictionResponse
from src.api.service import MLService

app = FastAPI(
    title="Student Mental Health Signal — Pro API",
    description="Production Machine Learning Microservice for Student Wellness Analytics & Explainable AI (XAI)",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Modular API Routers
app.include_router(health.router, prefix="/api/v1", tags=["System Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Inference"])
app.include_router(explain.router, prefix="/api/v1", tags=["Explainability"])
app.include_router(simulate.router, prefix="/api/v1", tags=["Simulation"])
app.include_router(cohort.router, prefix="/api/v1", tags=["Cohort Analytics"])

# Backward-compatible root /predict endpoint matching original API contract
@app.post('/predict', response_model=dict, tags=["Legacy Support"])
def legacy_predict(data: StudentInputSchema):
    service = MLService.get_instance()
    res = service.predict(data)
    return {"predicted_mental_health_score": res.predicted_score}

# Serve Frontend static assets if available
STATIC_DIR = os.path.dirname(__file__)
index_path = os.path.join(STATIC_DIR, "index.html")
css_path = os.path.join(STATIC_DIR, "style.css")
js_path = os.path.join(STATIC_DIR, "script.js")

if os.path.exists(index_path):
    @app.get('/', include_in_schema=False)
    def serve_frontend():
        return FileResponse(index_path)

if os.path.exists(css_path):
    @app.get('/style.css', include_in_schema=False)
    def serve_css():
        return FileResponse(css_path, media_type="text/css")

if os.path.exists(js_path):
    @app.get('/script.js', include_in_schema=False)
    def serve_js():
        return FileResponse(js_path, media_type="application/javascript")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main.py:app", host="0.0.0.0", port=8000, reload=True)
