VLLM_OMNI_USE_V1=1 VLLM_LOGGING_LEVEL=INFO python text_to_image_async.py \
    --model Tongyi-MAI/Z-Image-Turbo \
    --prompt "a cup of coffee on the table" \
    --num_inference_steps 8 \
    --output output.png
