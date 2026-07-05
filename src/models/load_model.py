import logging

from diffusers import AutoencoderKLQwenImage

from src.models.transformer_qwenimage import QwenImageTransformer2DModel


def load_vae(pretrained_model_name_or_path, weight_dtype):
    vae = AutoencoderKLQwenImage.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
        use_safetensors=True,  # 使用 safetensors 格式，加载更快
        device_map="cpu",
    )
    logging.info(f"loaded vae from {pretrained_model_name_or_path} with weight_dtype {weight_dtype}")
    return vae


def load_qwenvl(pretrained_model_name_or_path, weight_dtype):
    from transformers import Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
        use_safetensors=True,
        attn_implementation="flash_attention_2",
    )
    logging.info(f"loaded qwen_vl from {pretrained_model_name_or_path} with weight_dtype {weight_dtype}")
    return model


def load_transformer(pretrained_model_name_or_path, weight_dtype, device_map="cuda:1"):
    logging.info(f"load model {pretrained_model_name_or_path}")
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        use_safetensors=True,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    logging.info(f"loaded transformer from {pretrained_model_name_or_path} with weight_dtype {weight_dtype}")
    return flux_transformer
