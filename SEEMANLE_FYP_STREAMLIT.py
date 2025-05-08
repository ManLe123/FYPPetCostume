import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageOps
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from diffusers import ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetInpaintPipeline

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Title
st.title("Object-Guided ControlNet Inpainting")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
prompt = st.text_input("Prompt", "add batman costume on this cat, high quality, natural")

if uploaded_file:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)

    # Resize
    max_size = 768
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Run detection
    text_labels = [["animal and head"]]
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=0.3, text_threshold=0.3, target_sizes=[image.size[::-1]]
    )

    boxes = results[0]["boxes"].cpu().numpy()
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    x_min_l, y_min_l, x_max_l, y_max_l = map(int, boxes[0])

    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    mask[y_min_l:y_max_l, x_min_l:x_max_l] = 255

    # Get canny
    image_np = np.array(image)
    canny = cv2.Canny(image_np, 100, 200)
    canny = np.stack([canny] * 3, axis=-1)
    canny_image = Image.fromarray(canny)
    mask_image = Image.fromarray(mask)

    # Load pipeline
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant="fp16"
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.to("cuda")

    if st.button("Generate"):
        generator = torch.manual_seed(0)
        output = pipe(
            prompt,
            num_inference_steps=20,
            generator=generator,
            image=image,
            control_image=canny_image,
            controlnet_conditioning_scale=0.1,
            image_guidance_scale=2.0,
            mask_image=mask_image,
        ).images[0]
        st.image(output, caption="Generated Image", use_column_width=True)