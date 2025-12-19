from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

model_path = "liuhaotian/llava-v1.5-7b"
model_name = get_model_name_from_path(model_path)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name
)

print("Model loaded successfully")
