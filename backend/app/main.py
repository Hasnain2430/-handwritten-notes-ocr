from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import sys
from datetime import datetime
from pathlib import Path

# Configure logging with detailed format - MUST BE BEFORE OTHER IMPORTS
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from backend directory
    env_path = Path(__file__).parent.parent / '.env'
    
    # Try loading with default encoding (usually utf-8)
    try:
        load_dotenv(env_path, encoding='utf-8')
        logger.info(f"✅ Loaded environment variables from: {env_path}")
    except UnicodeDecodeError:
        logger.warning(f"⚠️  UTF-8 decode failed for .env, trying utf-16...")
        try:
            load_dotenv(env_path, encoding='utf-16')
            logger.info(f"✅ Loaded environment variables (utf-16) from: {env_path}")
        except Exception:
            logger.warning(f"⚠️  UTF-16 decode failed for .env, trying latin-1...")
            load_dotenv(env_path, encoding='latin-1')
            logger.info(f"Loaded environment variables from: {env_path}")
    except Exception as e:
        if "null character" in str(e):
            logger.warning("  Detected null characters in .env (likely UTF-16), trying utf-16...")
            load_dotenv(env_path, encoding='utf-16')
            logger.info(f"Loaded environment variables from: {env_path}")
        else:
            raise e

    # DEBUG: Check if ALIBABA_API_KEY is loaded
    import os
    ALIBABA_API_KEY = os.getenv('ALIBABA_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
    QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-vl-ocr")

    if not ALIBABA_API_KEY:
        logger.error("ALIBABA_API_KEY or DASHSCOPE_API_KEY NOT FOUND in environment variables")
        logger.error(f"   Current working directory: {os.getcwd()}")
        logger.error(f"   .env path exists: {env_path.exists()}")
        if env_path.exists():
            # Try reading with utf-8, fallback to utf-16, then latin-1
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(env_path, 'r', encoding='utf-16') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(env_path, 'r', encoding='utf-16') as f:
                            content = f.read()
                    except UnicodeError:
                        with open(env_path, 'r', encoding='latin-1') as f:
                            content = f.read()
                
                logger.info(f"   .env content length: {len(content)}")
                if 'ALIBABA_API_KEY' in content:
                    logger.info("   'ALIBABA_API_KEY' string found in .env file")
                else:
                    logger.error("   'ALIBABA_API_KEY' string NOT found in .env file")

except ImportError:
    # python-dotenv not installed - skip .env loading
    logger.warning("python-dotenv not installed - .env file will NOT be loaded")
except Exception as e:
    # .env file not found or error loading - continue without it
    logger.error(f"Error loading .env file: {e}")

from app.api import agent_convert, convert, upload, download

logger = logging.getLogger(__name__)

app = FastAPI(title="Handwritten Notes OCR to Word")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(convert.router, prefix="/api", tags=["convert"])
app.include_router(agent_convert.router, prefix="/api", tags=["agent"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(download.router, prefix="/api", tags=["download"])

@app.on_event("startup")
async def startup_event():
    """Log server startup information."""
    import os
    port = os.getenv("PORT", "8000")
    logger.info("=" * 60)
    logger.info("Handwritten Notes OCR API Server Starting...")
    logger.info("=" * 60)
    logger.info(f"Server started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Server will be available at: http://0.0.0.0:{port}")
    logger.info(f"API Documentation: http://0.0.0.0:{port}/docs")
    logger.info(f"Health Check: http://0.0.0.0:{port}/")
    logger.info("=" * 60)
    logger.info("NOTE: ML models load lazily on first request to save memory")
    logger.info("=" * 60)

@app.get("/")
async def root():
    logger.info("Health check endpoint accessed")
    return {"status": "ok", "message": "Handwritten Notes OCR API"}

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting server with uvicorn on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
