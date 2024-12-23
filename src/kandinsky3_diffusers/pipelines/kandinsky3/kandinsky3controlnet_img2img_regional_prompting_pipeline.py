import html
import inspect
import re
import urllib.parse as ul
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import CLIPImageProcessor, T5EncoderModel, T5Tokenizer

from ...loaders import LoraLoaderMixin
from ...models import UNetKandi3, VQModel, UNetKandi3Controlnet
from ...schedulers import DDPMScheduler 
from ...utils import (
    BACKENDS_MAPPING,
    is_accelerate_available,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from .kandinsky3controlnet_img2img_pipeline import KandinskyV3ControlnetImg2imgPipeline
import math
from einops import rearrange, repeat
from torch import einsum
import PIL

def exist(item):
    return item is not None

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class KandinskyV3ControlnetImg2imgRegionalPromptingPipeline(KandinskyV3ControlnetImg2imgPipeline):

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        unet: UNetKandi3,
        scheduler: DDPMScheduler,
        movq: VQModel,
        unet_controlnet: UNetKandi3Controlnet
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            movq=movq,
            unet_controlnet=unet_controlnet,
        )
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            movq=movq,
            unet_controlnet=unet_controlnet,
        )
        
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
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
        guidance_rescale: float = 0.0,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        latents=None,
        cut_context: bool = False,
        ### image2image
        image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        strength: float = 0.3,
        ### ControlNet
        hint=None,
        k = 1,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        ### Regional Prompting
        regional_prompts: List[str] = None,
        regional_negative_prompts: List[str] = None,
        regional_masks: List[torch.Tensor] = None,
        base_ratio: float = 0.0,
        regional_k: List[float] = None,
    ):  

        device = self._execution_device
        regions = len(regional_masks)
        orig_hw = (height, width)
        
        (
            regional_prompt_embeds, 
            _, 
            regional_prompt_embeds_mask, 
            _, 
            _
        ) = self.encode_prompt(
            prompt=regional_prompts,
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
            prompt=regional_negative_prompts,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        
        def hook_forward(module):
            
            def forward(
                x: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None,
                image_mask: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
                
                attn = module
                xshape = x.shape # (2, h * w, emb_dim), e.g. h = w = 64, emb_dim = 768
                (h, w) = calculate_dimensions(xshape[1], *orig_hw)
                
                # expand x to match number of regions
                x_uncond, x_text = x.chunk(2)
                x = torch.cat(
                    [x_uncond, x_text] + regions * [x_uncond] + regions * [x_text], 
                    dim=0
                )

                # expand context to match x shape
                context = torch.cat(
                    [context, regional_negative_prompt_embeds, regional_prompt_embeds],
                    dim=0
                )
                context_mask = torch.cat(
                    [context_mask, regional_negative_prompt_embeds_mask, regional_prompt_embeds_mask],
                    dim=0
                )

                query = rearrange(attn.to_query(x), 'b n (h d) -> b h n d', h=attn.num_heads)
                key = rearrange(attn.to_key(context), 'b n (h d) -> b h n d', h=attn.num_heads)
                value = rearrange(attn.to_value(context), 'b n (h d) -> b h n d', h=attn.num_heads)
                attention_matrix = einsum('b h i d, b h j d -> b h i j', query, key)
                
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
                reshaped = x.reshape(x.size()[0], h, w, x.size()[2])
                x_base = reshaped[:2]
                x_uncond, x_text = reshaped[2:].chunk(2)
                for out in (x_uncond, x_text):
                    for region_idx, regional_mask in enumerate(regional_masks):
                        regional_mask = torch.nn.functional.interpolate(regional_mask[None, None, :, :], (h, w), mode='nearest-exact')
                        regional_mask = regional_mask.permute(0, 2, 3, 1).to(x.device)
                        # Replace masked area of the latent with the corresponding regional latent
                        out[0] = regional_mask * out[region_idx:region_idx+1] + (1 - regional_mask) * out[0]
                x = torch.stack([x_uncond[0], x_text[0]], dim=0)
                
                # apply base ratio
                x = (1 - base_ratio) * x + base_ratio * x_base
                x = x.reshape(xshape)
                    
                return x

            return forward

        def hook_forwards(root_module: torch.nn.Module):
            for name, module in root_module.named_modules():
                if "attn" in name and module.__class__.__name__ == "Attention":
                    module.forward = hook_forward(module)

        hook_forwards(self.unet)
            
        if regional_k is not None:
            k = torch.zeros((height, width), device=device)
            for cond_scale, regional_mask in zip(regional_k, regional_masks):
                k += cond_scale * regional_mask.to(device)        
            
        output = KandinskyV3ControlnetImg2imgPipeline(**self.components)(
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
            guidance_rescale=guidance_rescale,
            num_images_per_prompt=num_images_per_prompt,
            cut_context=cut_context,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type=output_type,
            return_dict=return_dict,
            hint=hint,
            k=k,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
        )

        return output


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