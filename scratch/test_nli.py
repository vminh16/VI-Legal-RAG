import logging
import numpy as np
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test MoritzLaurer model
model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
logger.info(f"Loading {model_name}...")

try:
    model = CrossEncoder(model_name)
    logger.info("Successfully loaded MoritzLaurer NLI model!")
    
    # Test premise and hypothesis in Vietnamese
    premise = "Người lao động làm việc thử việc tối đa không quá 30 ngày đối với lao động chuyên môn nghiệp vụ trung cấp."
    hypothesis = "Thời gian thử việc tối đa của trung cấp là 30 ngày."
    
    scores = model.predict([premise, hypothesis])
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()
    
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    if hasattr(model, "model") and hasattr(model.model, "config") and hasattr(model.model.config, "id2label"):
        label_map = {int(k): str(v).lower() for k, v in model.model.config.id2label.items()}
        
    prob_dict = {label_map.get(i, f"label_{i}"): float(p) for i, p in enumerate(probs)}
    logger.info(f"Results: {prob_dict}")
except Exception as e:
    logger.error(f"Failed: {e}")
