from inference_solver import FlexARInferenceSolver
from PIL import Image

# ******************** Image Generation ********************
inference_solver = FlexARInferenceSolver(
    model_path="Alpha-VLLM/Lumina-mGPT-7B-768",
    precision="bf16",
    target_size=768,
)

q1 = f"Generate an image of 768x768 according to the following prompt:\nImage of a dog playing water, and a waterfall is in the background."

# generated: tuple of (generated response, list of generated images)
generated = inference_solver.generate(
    images=[],
    qas=[[q1, None]],
    max_gen_len=8192,
    temperature=1.0,
    logits_processor=inference_solver.create_logits_processor(cfg=4.0, image_top_k=2000),
)

a1, new_image = generated[0], generated[1][0]

new_image.save("generated_image.png")

# # ******************* Image Understanding ******************
# inference_solver = FlexARInferenceSolver(
#     model_path="Alpha-VLLM/Lumina-mGPT-7B-512",
#     precision="bf16",
#     target_size=512,
# )

# # "<|image|>" symbol will be replaced with sequence of image tokens before fed to LLM
# q1 = "Describe the image in detail. <|image|>"

# images = [Image.open("/home/kris/generation_for_compression/dataset/image_kodak/kodim04.png")]
# qas = [[q1, None]]

# # `len(images)` should be equal to the number of appearance of "<|image|>" in qas
# generated = inference_solver.generate(
#     images=images,
#     qas=qas,
#     max_gen_len=8192,
#     temperature=1.0,
#     logits_processor=inference_solver.create_logits_processor(cfg=4.0, image_top_k=2000),
# )

# a1 = generated[0]
# # generated[1], namely the list of newly generated images, should typically be empty in this case.


# # ********************* Omni-Potent *********************
# inference_solver = FlexARInferenceSolver(
#     model_path="Alpha-VLLM/Lumina-mGPT-7B-768-Omni",
#     precision="bf16",
#     target_size=768,
# )

# # Example: Depth Estimation
# # For more instructions, see demos/demo_image2image.py
# q1 = "Depth estimation. <|image|>"
# images = [Image.open("image.png")]
# qas = [[q1, None]]

# generated = inference_solver.generate(
#     images=images,
#     qas=qas,
#     max_gen_len=8192,
#     temperature=1.0,
#     logits_processor=inference_solver.create_logits_processor(cfg=1.0, image_top_k=200),
# )

# a1 = generated[0]
# new_image = generated[1][0]
