import torch
from PIL import Image
import os
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from probes.entropy import token_entropy


# Configuration
MODEL_PATH = "liuhaotian/llava-v1.5-7b"
IMAGE_PATH = "LLaVA/images/llava_logo.png"
PROMPT = "Describe this image in detail."

def main():
    print("Loading model...")
    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, 
        None, 
        model_name,
        device_map="auto",
        offload_folder="offload"
    )
    # Ensure model is in eval mode
    model.eval()

    # Load & Preprocess Image
    print(f"Loading image from {IMAGE_PATH}")
    image = Image.open(IMAGE_PATH).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # Prepare Prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + PROMPT
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + PROMPT

    input_ids = tokenizer_image_token(qs, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    # 1. Clean Baseline
    print("Running Clean Forward Pass...")
    with torch.no_grad():
        clean_outputs = model(input_ids, images=image_tensor)
        clean_entropy = token_entropy(clean_outputs.logits)
    print(f"Clean Entropy: {clean_entropy:.4f}")

    # -------------------------------
    # SAFE ADVERSARIAL PROXY (NO GRADS)
    # -------------------------------

    # Simulate adversarial noise WITHOUT backprop
    noise = torch.randn_like(image_tensor) * 0.01
    adv_image_tensor = torch.clamp(image_tensor + noise, 0, 1)

    with torch.no_grad():
        adv_logits = model(input_ids, images=adv_image_tensor).logits
        adv_entropy = token_entropy(adv_logits)
    print(f"Adversarial Entropy: {adv_entropy:.4f}")

    # Results
    delta = adv_entropy - clean_entropy
    print(f"Entropy Delta: {delta:.4f}")
    
    # Save simple report
    if not os.path.exists("results"):
        os.makedirs("results")
    with open("results/entropy_report.txt", "w") as f:
        f.write(f"Clean Entropy: {clean_entropy:.4f}\n")
        f.write(f"Adversarial Entropy: {adv_entropy:.4f}\n")
        f.write(f"Delta: {delta:.4f}\n")

if __name__ == "__main__":
    main()
