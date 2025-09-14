from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import generate, datasets, auth, schemas, evaluation, generate_fast, generate_optimized, exports, advanced_types
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Synthetic Data Exchange",
    description="A platform for generating and sharing synthetic datasets",
    version="0.2.0"
)

# CORS configuration
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(generate.router, tags=["Generation"])
app.include_router(generate_fast.router, tags=["Fast Generation"])
app.include_router(generate_optimized.router, tags=["Optimized Generation"])
app.include_router(datasets.router, tags=["Datasets"])
app.include_router(auth.router, tags=["Authentication"])
app.include_router(schemas.router, prefix="/schemas", tags=["Schema Management"])
app.include_router(evaluation.router, prefix="/evaluation", tags=["Quality Evaluation"])
app.include_router(exports.router, prefix="/export", tags=["Export Formats"])
app.include_router(advanced_types.router, prefix="/advanced", tags=["Advanced Data Types"])

@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Welcome to Synthetic Data Exchange",
        "version": "0.2.0",
        "docs": "/docs",
        "status": "healthy"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": "synthetic-data-exchange"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
