import html
import inspect
import re
import urllib.parse as ul
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import CLIPImageProcessor, T5EncoderModel, T5Tokenizer

from ...loaders import LoraLoaderMixin
from ...models import UNetKandi3, VQModel
from ...schedulers import DDPMScheduler 
from ...utils import (
    BACKENDS_MAPPING,
    is_accelerate_available,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from .kandinsky3img2img_pipeline import KandinskyV3Img2ImgPipeline
import math
from einops import rearrange, repeat
from torch import einsum
import PIL
from PIL import Image

def exist(item):
    return item is not None

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

KEYWORD_COMMON_PROMPT = "ADDCOMM"
KEYWORD_BREAK = "BREAK"


class KandinskyV3Img2ImgRegionalPromptingPipeline(KandinskyV3Img2ImgPipeline):

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        unet: UNetKandi3,
        scheduler: DDPMScheduler,
        movq: VQModel
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            movq=movq
        )
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            movq=movq
        )
        
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        strength: float = 0.3,
        num_inference_steps: int = 100,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        latents=None,
        cut_context=False,
        ### Regional Prompting
        rp_kwargs: Dict[str, str] = None,
    ):  

        active = KEYWORD_BREAK in prompt[0] if isinstance(prompt, list) else KEYWORD_BREAK in prompt
        if negative_prompt is None:
            negative_prompt = "" if isinstance(prompt, str) else [""] * len(prompt)
            
        device = self._execution_device
        regions = 0
            
        prompts = prompt if isinstance(prompt, list) else [prompt]
        negative_prompts = negative_prompt if isinstance(prompt, list) else [negative_prompt] # HERE I CHANGED `str` TO `list`
        self.batch = batch = num_images_per_prompt * len(prompts) # = 1 almost always
        prompts_concatenated = create_prompts_for_regions(prompts, num_images_per_prompt)
        negative_prompts_concatenated = create_prompts_for_regions(negative_prompts, num_images_per_prompt)
        equal = len(prompts_concatenated) == len(negative_prompts_concatenated)
        
        (
            regional_prompt_embeds, 
            _, 
            regional_prompt_embeds_mask, 
            _, 
            _
        ) = self.encode_prompt(
            prompt=prompts_concatenated,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        
        (
            regional_negative_prompt_embeds, 
            _, 
            regional_negative_prompt_embeds_mask, 
            _, 
            _
        ) = self.encode_prompt(
            prompt=negative_prompts_concatenated,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        
        prompt_embeds = negative_prompt_embeds = None
        
        if active:
                
            regional_masks = rp_kwargs["regional_masks"]
            regions = len(regional_masks)

            orig_hw = (height, width)
            revers = True

            def hook_forward(module):
                
                def forward(
                    x: torch.Tensor,
                    context: Optional[torch.Tensor] = None,
                    context_mask: Optional[torch.Tensor] = None,
                    image_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
                    
                    attn = module
                    xshape = x.shape # (2, h * w, emb_dim), e.g. h = w = 64, emb_dim = 768
                    self.hw = (h, w) = calculate_dimensions(xshape[1], *orig_hw)

                    if revers:
                        x_uncond, x_text = x.chunk(2)
                    else:
                        x_text, x_uncond = x.chunk(2)
                        
                    # print('regions:', regions)

                    if equal:
                        x = torch.cat(
                            [x_text for i in range(regions)] + [x_uncond for i in range(regions)],
                            0,
                        )
                    else:
                        x = torch.cat([x_text for i in range(regions)] + [x_uncond], 0)
                        
                    # x: (1 + regions, h * w, emb_dim), regions = 3, h = w = 64, emb_dim = 768
                    # regional_prompt_embeds: (regions, seq_len, context_dim), e.g. regions = 3, seq_len = 128, context_dim = 4096
                    # regional_negative_prompt_embeds: (1, seq_len, context_dim), e.g. regions = 3, seq_len = 128, context_dim = 4096
                    context = torch.cat([regional_prompt_embeds] + [regional_negative_prompt_embeds], 0)
                    context_mask = torch.cat([regional_prompt_embeds_mask] + [regional_negative_prompt_embeds_mask], 0)

                    query = rearrange(attn.to_query(x), 'b n (h d) -> b h n d', h=attn.num_heads)
                    key = rearrange(attn.to_key(context), 'b n (h d) -> b h n d', h=attn.num_heads)
                    value = rearrange(attn.to_value(context), 'b n (h d) -> b h n d', h=attn.num_heads)
                    attention_matrix = einsum('b h i d, b h j d -> b h i j', query, key) # (1 + regions, num_heads, h * w, seq_len), e.g. regions = 3, num_heads = 24, h = w = 64, seq_len = 128

                    if exist(image_mask) and exist(context_mask):
                        image_mask = rearrange(image_mask, 'b i -> b 1 i 1')
                        image_text_mask_1 = rearrange((context_mask == 1).type(image_mask.dtype), 'b j -> b 1 1 j')
                        image_text_mask_2 = rearrange((context_mask == 2).type(image_mask.dtype), 'b j -> b 1 1 j')
                        
                        image_mask_max = image_mask.amax(-2, keepdim=True)
                        max_attention = rearrange(attention_matrix.amax((-2, -1)), 'b h -> b h 1 1')
                        attention_matrix = attention_matrix + max_attention * (image_mask * image_text_mask_1)
                        attention_matrix = attention_matrix + max_attention * ((image_mask_max - image_mask) * image_text_mask_2)
                        
                    if exist(context_mask):
                        max_neg_value = -torch.finfo(attention_matrix.dtype).max
                        context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
                        attention_matrix = attention_matrix.masked_fill(~(context_mask != 0), max_neg_value)
                    attention_matrix = (attention_matrix * attn.scale).softmax(dim=-1)

                    x = einsum('b h i j, b h j d -> b h i d', attention_matrix, value)
                    x = rearrange(x, 'b h n d -> b n (h d)')
                    x = attn.output_layer(x)
                    
                    # Regional masked attention
                    reshaped = x.reshape(x.size()[0], h, w, x.size()[2]) # (1 + regions, h, w, emb_dim), e.g. regions = 3, h = w = 64, emb_dim = 768
                    center = reshaped.shape[0] // 2 # e.g. for regions = 3 -> center = 2
                    x_text = reshaped[0:center] if equal else reshaped[0:-batch] # e.g. first regions (if not equal)
                    x_uncond = reshaped[center:] if equal else reshaped[-batch:] # e.g. the last one (if not equal)
                    outs = [x_text, x_uncond] if equal else [x_text] # e.g. [(regions, h, w, emb_dim)]
                    for out in outs:
                        for region_idx, regional_mask in enumerate(regional_masks):
                            regional_mask = torch.nn.functional.interpolate(regional_mask[None, None, :, :], (h, w), mode='nearest-exact')
                            regional_mask = regional_mask.permute(0, 2, 3, 1).to(x.device)
                            # Replace masked area of the latent with the corresponding regional latent
                            out[0:batch] = regional_mask * out[region_idx * batch : (region_idx + 1) * batch] + (1 - regional_mask) * out[0:batch]
                    x_text, x_uncond = (x_text[0:batch], x_uncond[0:batch]) if equal else (x_text[0:batch], x_uncond)
                    x = torch.cat([x_uncond, x_text], 0) if revers else torch.cat([x_text, x_uncond], 0)
                    x = x.reshape(xshape)
                        
                    return x

                return forward

            def hook_forwards(root_module: torch.nn.Module):
                for name, module in root_module.named_modules():
                    if "attn" in name and module.__class__.__name__ == "Attention":
                        module.forward = hook_forward(module)

            hook_forwards(self.unet)
            
        output = KandinskyV3Img2ImgPipeline(**self.components)(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt=negative_prompt,
            negative_prompt_embeds=negative_prompt_embeds,
            image=image,
            strength=strength,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            cut_context=cut_context,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type=output_type,
            return_dict=return_dict,
        )

        return output
        
        
def create_prompts_for_regions(prompts, batch_size):
    """
    Creates a list of prompts for each region, considering common prompts and batch size.

    Args:
        prompts: A list of prompts, potentially containing common prompts and region-specific prompts.
        batch_size: The desired batch size for the prompts.

    Returns:
        A concatenated list of prompts ready for batch processing.
    """

    processed_prompts = []
    for prompt in prompts:
        prompt = prompt.strip()
        common_prompt, region_prompts = prompt.split(KEYWORD_COMMON_PROMPT) if KEYWORD_COMMON_PROMPT in prompt else ("", prompt)
        region_prompts = [f"{common_prompt.strip()} {p.strip()}" for p in region_prompts.split(KEYWORD_BREAK)]
        processed_prompts.append(region_prompts)

    concatenated_prompts = []
    for region_prompts in processed_prompts:
        for prompt in region_prompts:
            concatenated_prompts.extend([prompt] * batch_size)

    return concatenated_prompts


def calculate_dimensions(total_pixels, height, width):
    """
    Calculates the optimal dimensions for a given number of pixels and image dimensions.

    Args:
        total_pixels: The total number of pixels to distribute.
        height: The original height of the image.
        width: The original width of the image.

    Returns:
        A tuple of the calculated height and width.
    """

    scale_factor = math.ceil(math.log2(math.sqrt(height * width / total_pixels)))
    new_height = math.ceil(height / 2**scale_factor)
    new_width = math.ceil(width / 2**scale_factor)

    return new_height, new_width