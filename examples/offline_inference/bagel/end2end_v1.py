"""
BAGEL end-to-end offline inference using the V1 engine.

text2img: stage-0 LLM (max_tokens=1) → stage-1 diffusion → PIL images
"""

import asyncio
import os
import uuid

from vllm.sampling_params import SamplingParams

from vllm_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ByteDance-Seed/BAGEL-7B-MoT")
    parser.add_argument("--prompts", nargs="+", default=None)
    parser.add_argument("--txt-prompts", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=1.5)
    parser.add_argument("--output-dir", type=str, default="output_images")
    parser.add_argument("--stage-init-timeout", type=int, default=300)
    args = parser.parse_args()

    # Load prompts
    if args.txt_prompts:
        with open(args.txt_prompts, encoding="utf-8") as f:
            raw_prompts = [ln.strip() for ln in f if ln.strip()]
    elif args.prompts:
        raw_prompts = args.prompts
    else:
        raw_prompts = ["A cute cat"]

    prompts = [{"prompt": f"<|im_start|>{p}<|im_end|>", "modalities": ["image"]} for p in raw_prompts]

    omni = AsyncOmni(model=args.model, stage_init_timeout=args.stage_init_timeout, log_stats=True)

    llm_params = SamplingParams(max_tokens=1)
    diffusion_params = OmniDiffusionSamplingParams(
        num_inference_steps=args.steps,
        extra_args={
            "cfg_text_scale": args.cfg_text_scale,
            "cfg_img_scale": args.cfg_img_scale,
        },
    )
    sampling_params_list = [llm_params, diffusion_params]

    os.makedirs(args.output_dir, exist_ok=True)

    async def process(prompt, request_id):
        async for stage_output in omni.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
        ):
            if stage_output.final_output_type == "image" and stage_output.images:
                for j, img in enumerate(stage_output.images):
                    path = os.path.join(args.output_dir, f"{request_id}_{j}.png")
                    img.save(path)
                    print(f"Saved {path}")

    tasks = [process(p, str(uuid.uuid4())) for p in prompts]
    await asyncio.gather(*tasks)

    omni.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
