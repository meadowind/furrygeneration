import os
# Configure PyTorch CUDA allocator to prevent fragmentation (helps with memory efficiency)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import random
import torch 
from datetime import datetime
from PIL import Image

# Globally disable cuDNN to resolve potential hardware convolution conflicts
torch.backends.cudnn.enabled = False

from diffusers import StableDiffusionXLPipeline

device = "cuda"
dtype = torch.float16

# ==========================================
# [Experiment Parameter Settings]
# ==========================================
# Fixed seed for reproducibility       Set to None if random generation is desired
CUSTOM_SEED = 20260731                

# File paths for the animals reference images used in the cross-group experiment
EXPERIMENT_GROUPS = {
    #"rabbit": "/home/han/final/Animal/rabbit.jpg",
    #"cat": "/home/han/final/Animal/cat.jpg",
    "bird": "/home/han/final/Animal/bird.jpg",
    #"bear": "/home/han/final/Animal/bear.jpg",
    #"redpanda": "/home/han/final/Animal/redpanda.jpg"
}

# Tailored Prompt and Negative Prompt dictionary for each animal group
ANIMAL_PROMPTS = {
    "rabbit": {
        "prompt": "masterpiece, soft natural lighting, detailed skin texture, same pose of the cute kemonomimi person with the rabbit ears seamlessly integrated with human hair, harmonious facial features, clean soft skin, cinematic composition, dark textured background",
        "negative_prompt": "separate animal, extra limbs, deformed face, bad anatomy, blurry, low quality, wrong face, wrong animal, changed identity, pointed ears, cat ears, sharp muzzle, fox ears"
    },
    "cat": {
        "prompt": "masterpiece, soft natural lighting, detailed skin texture, portrait of a cute kemonomimi person with pointed cat ears and subtle cat-like features seamlessly integrated with human hair, harmonious facial features, soft fur texture, cinematic composition, dark textured background",
        "negative_prompt": "separate animal, extra limbs, deformed face, bad anatomy, blurry, low quality, wrong face, wrong animal, changed identity, floppy ears, round ears, fox ears"
    },
    "bird": {
        "prompt": "masterpiece, soft natural lighting, detailed skin texture, portrait of a cute kemonomimi person with delicate feathered hair ornaments and subtle avian characteristics seamlessly integrated, harmonious facial features,bird wings at side, dark textured background",
        "negative_prompt": "separate animal, extra limbs, deformed face, bad anatomy, blurry, low quality, wrong face, wrong animal, changed identity, animal ears, fur, snout"
    },
    "bear": {
        "prompt": "masterpiece, soft natural lighting, detailed skin texture, portrait of a cute kemonomimi person with small round bear ears seamlessly integrated with human hair, harmonious facial features, thick brown fur texture naturally blended on face, cinematic composition, dark textured background",
        "negative_prompt": "separate animal, extra limbs, deformed face, bad anatomy, blurry, low quality, wrong face, wrong animal, changed identity, pointed ears, cat ears, fox ears"
    },
    "redpanda": {
        "prompt": "masterpiece, soft natural lighting, detailed skin texture, portrait of a cute kemonomimi person with fluffy red panda ears and soft reddish-brown tail accents seamlessly integrated with human hair, harmonious facial features, soft fur texture, cinematic composition, dark textured background",
        "negative_prompt": "separate animal, extra limbs, deformed face, bad anatomy, blurry, low quality, wrong face, wrong animal, changed identity, pointed ears, cat ears, fox ears"
}
}

# weight gradient interpolation steps (animal_adapter_weight, human_adapter_weight)
# Higher value indicates stronger influence from the respective reference image.
WEIGHT_STEPS = [
    (0.4,0.6)
]
# ==========================================

print("=== [Step 1] Initialize SDXL Base Model ===")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=dtype,
    use_safetensors=True
)

print("=== [Step 2] Load Dual IP-Adapter Modules (Animal + Face Fusion) ===")
pipe.load_ip_adapter(
    "h94/IP-Adapter", 
    subfolder="sdxl_models", 
    weight_name=[
        "ip-adapter_sdxl.safetensors",
        "ip-adapter_sdxl.safetensors"
    ],
    adapter_name=["animal_adapter", "face_adapter"]
)

# Enable CPU offloading to optimize VRAM usage for dual-adapter pipeline
pipe.enable_model_cpu_offload()

# Enable VAE slicing and tiling to prevent Out-Of-Memory (OOM) errors during decoding
if hasattr(pipe, "vae") and pipe.vae is not None:
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

print("=== [Step 3] Preprocess Human Reference Image (Black Background Padding & Resizing) ===")
human_img_path = "/home/han/final/Human/01.jpg"  #path of the human reference image

def black_pad_to_square(image_path, target_size=512, background_color=(0, 0, 0)):
    """Resizes and pads an image with a black background to make it square without distortion."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    max_dim = max(width, height)
    new_img = Image.new("RGB", (max_dim, max_dim), background_color)
    paste_x = (max_dim - width) // 2
    paste_y = (max_dim - height) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

human_img = black_pad_to_square(human_img_path, 512)
human_filename = os.path.basename(human_img_path)
save_name = f"processed_human_{human_filename}"
human_img.save(save_name)
print(f" Preprocessed human image saved as: {save_name}")

# Determine the random seed value
if CUSTOM_SEED is None:
    current_seed = random.randint(0, 2147483647)
else:
    current_seed = CUSTOM_SEED

# 🚀 Start automated cross-group batch generation loop
for group_name, animal_path in EXPERIMENT_GROUPS.items():
    print(f"\n========================================")
    print(f" Processing animal group: [ {group_name.upper()} ]")
    print(f"========================================")
    
    # Retrieve group-specific prompts, defaulting to 'rabbit' if not found
    current_prompt = ANIMAL_PROMPTS.get(group_name, ANIMAL_PROMPTS["rabbit"])["prompt"]
    current_neg_prompt = ANIMAL_PROMPTS.get(group_name, ANIMAL_PROMPTS["rabbit"])["negative_prompt"]
    
    output_dir = f"Result_{group_name}"
    processed_dir = f"ProcessedReference_{group_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Preprocess and save the current animal reference image
    if os.path.exists(animal_path):
        animal_img = black_pad_to_square(animal_path, 512)
        animal_img.save(os.path.join(processed_dir, f"crop_{os.path.basename(animal_path)}"))
    else:
        print(f" Warning: Animal image path not found: {animal_path}. Skipping this group.")
        continue
        
    log_file = os.path.join(output_dir, "experiment_log.json")
    logs = []

    # Iterate through all weight configuration steps
    for idx, (animal_w, human_w) in enumerate(WEIGHT_STEPS, 1):
        print(f"[{group_name}] Progress [{idx}/{len(WEIGHT_STEPS)}] -> Animal Weight: {animal_w} | Human Weight: {human_w}")
        
        # Dynamically adjust the scales for both IP-Adapters
        pipe.set_ip_adapter_scale([animal_w, human_w])
        
        generator = torch.Generator(device="cpu").manual_seed(current_seed)
        torch.cuda.empty_cache()
        
        try:
            # Execute diffusion pipeline inference
            result = pipe(
                prompt=current_prompt,
                negative_prompt=current_neg_prompt,
                ip_adapter_image=[[animal_img], [human_img]], 
                num_inference_steps=15,
                height=512,
                width=512,
                generator=generator
            ).images[0]

            # Save generated fusion result
            output_filename = os.path.join(output_dir, f"fusion_animal_{animal_w}_human_{human_w}.jpg")
            result.save(output_filename)

            # Record experiment metadata for logging
            log_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": output_filename,
                "animal_group": group_name,
                "seed": current_seed,
                "weights": {"animal_adapter": animal_w, "human_adapter": human_w},
                "prompt": current_prompt,
                "negative_prompt": current_neg_prompt
            }
            logs.append(log_data)

        except Exception as e:
            print(f" Error occurred during execution: {e}")

    # Export experiment logs to JSON format
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)
    print(f" [ {group_name.upper()} ] group completed. Results saved to {output_dir}/")

# print("\n All 5 animal experiment groups have been successfully processed!")