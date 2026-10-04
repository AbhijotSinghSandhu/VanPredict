"""Production server entry point; supports the hosting provider's PORT."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.production:app", host=os.getenv("HOST", "0.0.0.0"),
                port=int(os.getenv("PORT", "8080")), workers=1)
