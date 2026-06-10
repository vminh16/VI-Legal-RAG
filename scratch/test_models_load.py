import sys
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("model_loader_test")

logger.info("Importing packages...")
start_time = time.time()

try:
    from src.pipeline.rag_pipeline import RAGPipeline
    logger.info(f"Imported in {time.time() - start_time:.2f} seconds.")
    
    logger.info("Initializing RAGPipeline (this will load all models)...")
    init_start = time.time()
    pipeline = RAGPipeline()
    logger.info(f"Initialized RAGPipeline successfully in {time.time() - init_start:.2f} seconds!")
    
except Exception as e:
    logger.error(f"Error during RAGPipeline load/init: {e}", exc_info=True)
    sys.exit(1)
