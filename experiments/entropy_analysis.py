import torch
from probes.entropy import token_entropy
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

model_path = "liuhaotian/llava-v1.5-7b"
model_name = get_model_name_from_path(model_path)

tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path, None, model_name
)

model.eval()

# TODO: load clean image and adversarial image
# TODO: run model forward pass
# TODO: compare entropy(clean) vs entropy(adversarial)
