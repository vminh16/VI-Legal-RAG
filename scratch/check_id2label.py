from sentence_transformers import CrossEncoder

model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
model = CrossEncoder(model_name)
print("id2label:", model.model.config.id2label)
