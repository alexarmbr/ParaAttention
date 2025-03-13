import torch
import torch.distributed as dist
from diffusers import WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from diffusers.utils import export_to_video
from safetensors.torch import load_file


def _patch_convert_non_diffusers_wan_lora_to_diffusers(state_dict):
    """existing HF lora loading code as of 3/12/24 assumes all keys are present, which is not the case & causes errors."""
    if not any(k.startswith("diffusion_model.") for k in state_dict):
        print("state dict is already in diffusers format")
        return

    converted_state_dict = {}
    original_state_dict = {k[len("diffusion_model.") :]: v for k, v in state_dict.items()}

    num_blocks = len({k.split("blocks.")[1].split(".")[0] for k in original_state_dict if "blocks." in k})

    def key_swap(converted_dict, orig_dict, converted_key, orig_key):
        if orig_key in orig_dict:
            converted_dict[converted_key] = orig_dict.pop(orig_key)

    for i in range(num_blocks):
        # Self-attention
        for o, c in zip(["q", "k", "v", "o"], ["to_q", "to_k", "to_v", "to_out.0"]):
            key_swap(
                converted_state_dict,
                original_state_dict,
                f"blocks.{i}.attn1.{c}.lora_A.weight",
                f"blocks.{i}.self_attn.{o}.lora_A.weight",
            )
            key_swap(
                converted_state_dict,
                original_state_dict,
                f"blocks.{i}.attn1.{c}.lora_B.weight",
                f"blocks.{i}.self_attn.{o}.lora_B.weight",
            )

        # Cross-attention
        for o, c in zip(["q", "k", "v", "o"], ["to_q", "to_k", "to_v", "to_out.0"]):
            key_swap(
                converted_state_dict,
                original_state_dict,
                f"blocks.{i}.attn2.{c}.lora_A.weight",
                f"blocks.{i}.cross_attn.{o}.lora_A.weight",
            )
            key_swap(
                converted_state_dict,
                original_state_dict,
                f"blocks.{i}.attn2.{c}.lora_B.weight",
                f"blocks.{i}.cross_attn.{o}.lora_B.weight",
            )
        for o, c in zip(["k_img", "v_img"], ["add_k_proj", "add_v_proj"]):
            key_swap(
                converted_state_dict,
                original_state_dict,
                f"blocks.{i}.attn2.{c}.lora_A.weight",
                f"blocks.{i}.cross_attn.{o}.lora_A.weight",
            )
            key_swap(
                converted_state_dict,
                original_state_dict,
                f"blocks.{i}.attn2.{c}.lora_B.weight",
                f"blocks.{i}.cross_attn.{o}.lora_B.weight",
            )

        # FFN
        for o, c in zip(["ffn.0", "ffn.2"], ["net.0.proj", "net.2"]):
            key_swap(
                converted_state_dict,
                original_state_dict,
                f"blocks.{i}.ffn.{c}.lora_A.weight",
                f"blocks.{i}.{o}.lora_A.weight",
            )
            key_swap(
                converted_state_dict,
                original_state_dict,
                f"blocks.{i}.ffn.{c}.lora_B.weight",
                f"blocks.{i}.{o}.lora_B.weight",
            )

    # Only raise an error if there are keys remaining that we expected to process
    lora_keys = [k for k in original_state_dict.keys() if "lora_A.weight" in k or "lora_B.weight" in k]
    if len(lora_keys) > 0:
        raise ValueError(f"Some LoRA keys couldn't be processed: {lora_keys}")

    for key in list(converted_state_dict.keys()):
        converted_state_dict[f"transformer.{key}"] = converted_state_dict.pop(key)

    return converted_state_dict


dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

# model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# flow shift should be 3.0 for 480p images, 5.0 for 720p images
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
pipe.to("cuda")

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

parallelize_pipe(
    pipe,
    mesh=init_context_parallel_mesh(
        pipe.device.type,
    ),
)

# Enable memory savings
# pipe.enable_model_cpu_offload(gpu_id=dist.get_rank())
# pipe.enable_vae_tiling()

# torch._inductor.config.reorder_for_compute_comm_overlap = True
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")


def generate(prompt, pipe, suffix):
    gen = torch.Generator(device=pipe.device).manual_seed(42)

    output = pipe(
        prompt=prompt,
        negative_prompt="",
        height=480,
        width=832,
        num_frames=81,
        num_inference_steps=30,
        generator=gen,
        output_type="pil" if dist.get_rank() == 0 else "pt",
    ).frames[0]

    if dist.get_rank() == 0:
        print(f"Saving video to wan_{suffix}.mp4")
        export_to_video(output, f"wan_{suffix}.mp4", fps=16)


# 'https://huggingface.co/motimalu/wan-flat-color-v2/resolve/main/wan_flat_color_v2.safetensors'
lora_one_file = "./loras/wan_flat_color_v2.safetensors"
lora_one_state_dict = _patch_convert_non_diffusers_wan_lora_to_diffusers(load_file(lora_one_file))

world_size = torch.distributed.get_world_size()

flat_prompt = "flat color 2d animation of a portrait of woman with white hair and green eyes, dynamic scene"
generate(flat_prompt, pipe, f"no_lora_{world_size}_gpu")

pipe.load_lora_weights(lora_one_state_dict)
generate(flat_prompt, pipe, f"flat_lora_{world_size}_gpu")

# pipe.unload_lora_weights()
# generate(flat_prompt, pipe, "no_flat_lora_again")

# # 'https://replicate.delivery/xezq/FbxX664a8aaoH1CLxHcM0lcegKStk0OaNCen0yufg2cGBovoA/trained_model.tar'
# lora_two_file = "./loras/sclera-lora/output/wan_train_replicate/lora.safetensors"
# lora_two_state_dict = _patch_convert_non_diffusers_wan_lora_to_diffusers(load_file(lora_two_file))

# sclera_prompt = "an extreme close up of an epic cyberpunk woman with BLACK_SCLERA"
# generate(sclera_prompt, pipe, "no_sclera_lora")

# pipe.load_lora_weights(lora_two_state_dict)
# generate(sclera_prompt, pipe, "sclera_lora")

# dist.destroy_process_group()
