import argparse
import os
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import pandas as pd
import math
from packaging import version

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from torchvision import transforms
import transformers
from transformers.utils import ContextManagers

import diffusers
from transformers import AutoTokenizer
from diffusers import DDPMScheduler, UNetKandi3, VQModel
from diffusers.optimization import get_scheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.models.attention_processor import LoraKandi3AttnProcessor
from omegaconf import OmegaConf
from copy import deepcopy
from transformers import T5EncoderModel
if is_wandb_available():
    import wandb
import torch
from torch.optim.optimizer import Optimizer

logger = get_logger(__name__, log_level="INFO")

def get_key_step(step, log_step_interval=50):
    step_d = step // log_step_interval
    return 'loss_' + str(step_d * log_step_interval) + '_' + str((step_d + 1) * log_step_interval)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of finetuning Kandinsky 3.")
    parser.add_argument(
        "--image_resolution",
        type=int,
        default=512,
        required=False,
        help="Image resolution",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        required=False,
        help="rank of the lora",
    )
    parser.add_argument(
        "--images_paths_csv",
        type=str,
        required=False,
        help="images_paths_csv",
    )
    
    parser.add_argument(
        "--pretrained_kandinsky_path",
        type=str,
        default='kandinsky-community/kandinsky-2-2-decoder',
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_path",
        type=str,
        default='kandinsky-community/kandinsky-2-2-decoder',
        required=False,
        help="Path to pretrained vae.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default='kandinsky-community/kandinsky-2-2-prior',
        required=False,
        help="Path to text_encoder.",
    )
    parser.add_argument(
        "--scheduler_path",
        type=str,
        default='kandinsky-community/kandinsky-2-2-decoder',
        required=False,
        help="Path to scheduler.",
    )
    parser.add_argument(
        "--uncondition_prob",
        type=float,
        default=0.1,
        required=False,
        help="train batch size",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        required=False,
        help="train batch size",
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        required=False,
        help="learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        required=False,
        help="weight decay",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="kandi_2_2-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_cache_emb",
        action="store_true",
        help=(
            ""
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.98, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def center_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images_paths_csv=None, tokenizer=None, image_resolution=512, use_cache_emb=False):
        self.use_cache_emb = use_cache_emb
        self.image_resolution = image_resolution
        self.tokenizer = tokenizer
        df = pd.read_csv(images_paths_csv)
        self.paths = df['paths'].values
        self.captions = df['caption'].values
        
    def set_cache_embeddings(self, text_masks, embeddings):
        self.text_masks = text_masks
        self.embeddings = embeddings
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, i):
        img = Image.open(self.paths[i])
        img = center_crop(img)
        img = img.resize((self.image_resolution, self.image_resolution), resample=Image.BICUBIC, reducing_gap=1)
        img = np.array(img.convert("RGB"))
        img = img.astype(np.float32) / 127.5 - 1
        img =  np.transpose(img, [2, 0, 1])
        if self.use_cache_emb:
            return img, self.embeddings[i], self.text_masks[i]
        else:
            text_inputs = self.tokenizer(
                self.captions[i],
                padding="max_length",
                max_length=128,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids[0]
            text_mask = text_inputs.attention_mask.bool()[0]
            return img, text_input_ids, text_mask

def process_embeds(embeddings, attention_mask):
    embeddings[attention_mask == 0] = torch.zeros_like(embeddings[attention_mask == 0])
    max_seq_length = attention_mask.sum(-1).max() + 1
    embeddings = embeddings[:, :max_seq_length]
    attention_mask = attention_mask[:, :max_seq_length]
    return embeddings, attention_mask

def encode( model, input_ids, attention_mask):
    with torch.no_grad():
        embeddings = model(input_ids, attention_mask=attention_mask)[0]
        #is_inf_embeddings = torch.isinf(embeddings).any(-1).any(-1)
        #is_nan_embeddings = torch.isnan(embeddings).any(-1).any(-1)
        #bad_embeddings = is_inf_embeddings + is_nan_embeddings
        #embeddings[bad_embeddings] = torch.zeros_like(embeddings[bad_embeddings])
    return embeddings


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
    noise_scheduler = DDPMScheduler.from_pretrained(args.scheduler_path, subfolder="scheduler")
    
    weight_dtype = torch.float32        
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16  
    vae = VQModel.from_pretrained(args.pretrained_vae_path, subfolder='movq', torch_dtype=weight_dtype).eval()
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_path, subfolder='text_encoder', torch_dtype=weight_dtype).eval()
    unet = UNetKandi3.from_pretrained(args.pretrained_kandinsky_path, subfolder="unet", torch_dtype=weight_dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_path, subfolder='tokenizer')

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    rank = args.lora_rank
    lora_attn_procs = {}
    lora_attn_procs['feature_pooling.attention.processor'] = LoraKandi3AttnProcessor(in_channels=4096, out_channels=1536, context_dim=4096, rank=rank)
    lora_attn_procs['down_samples.1.self_attention_block.attention.processor'] = LoraKandi3AttnProcessor(in_channels=384, out_channels=384, context_dim=384, rank=rank)
    lora_attn_procs['down_samples.1.resnet_attn_blocks.0.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=768, out_channels=768, context_dim=4096, rank=rank)
    lora_attn_procs['down_samples.1.resnet_attn_blocks.1.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=768, out_channels=768, context_dim=4096, rank=rank)
    lora_attn_procs['down_samples.1.resnet_attn_blocks.2.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=768, out_channels=768, context_dim=4096, rank=rank)
    lora_attn_procs['down_samples.2.self_attention_block.attention.processor'] = LoraKandi3AttnProcessor(in_channels=768, out_channels=768, context_dim=768, rank=rank)
    lora_attn_procs['down_samples.2.resnet_attn_blocks.0.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=1536, out_channels=1536, context_dim=4096, rank=rank)
    lora_attn_procs['down_samples.2.resnet_attn_blocks.1.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=1536, out_channels=1536, context_dim=4096, rank=rank)
    lora_attn_procs['down_samples.2.resnet_attn_blocks.2.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=1536, out_channels=1536, context_dim=4096, rank=rank)
    lora_attn_procs['down_samples.3.self_attention_block.attention.processor'] = LoraKandi3AttnProcessor(in_channels=1536, out_channels=1536, context_dim=1536, rank=rank)
    lora_attn_procs['down_samples.3.resnet_attn_blocks.0.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=3072, out_channels=3072, context_dim=4096, rank=rank)
    lora_attn_procs['down_samples.3.resnet_attn_blocks.1.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=3072, out_channels=3072, context_dim=4096, rank=rank)
    lora_attn_procs['down_samples.3.resnet_attn_blocks.2.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=3072, out_channels=3072, context_dim=4096, rank=rank)
    lora_attn_procs['up_samples.0.resnet_attn_blocks.0.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=3072, out_channels=3072, context_dim=4096, rank=rank)
    lora_attn_procs['up_samples.0.resnet_attn_blocks.1.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=3072, out_channels=3072, context_dim=4096, rank=rank)
    lora_attn_procs['up_samples.0.resnet_attn_blocks.2.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=3072, out_channels=3072, context_dim=4096, rank=rank)
    lora_attn_procs['up_samples.0.self_attention_block.attention.processor'] = LoraKandi3AttnProcessor(in_channels=1536, out_channels=1536, context_dim=1536, rank=rank)
    lora_attn_procs['up_samples.1.resnet_attn_blocks.0.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=3072, out_channels=3072, context_dim=4096, rank=rank)
    lora_attn_procs['up_samples.1.resnet_attn_blocks.1.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=1536, out_channels=1536, context_dim=4096, rank=rank)
    lora_attn_procs['up_samples.1.resnet_attn_blocks.2.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=1536, out_channels=1536, context_dim=4096, rank=rank)
    lora_attn_procs['up_samples.1.self_attention_block.attention.processor'] = LoraKandi3AttnProcessor(in_channels=768, out_channels=768, context_dim=768, rank=rank)
    lora_attn_procs['up_samples.2.resnet_attn_blocks.0.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=1536, out_channels=1536, context_dim=4096, rank=rank)
    lora_attn_procs['up_samples.2.resnet_attn_blocks.1.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=768, out_channels=768, context_dim=4096, rank=rank)
    lora_attn_procs['up_samples.2.resnet_attn_blocks.2.1.attention.processor'] = LoraKandi3AttnProcessor(in_channels=768, out_channels=768, context_dim=4096, rank=rank)
    lora_attn_procs['up_samples.2.self_attention_block.attention.processor'] = LoraKandi3AttnProcessor(in_channels=384, out_channels=384, context_dim=384, rank=rank)
    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
        
    optimizer = optimizer_cls(
        lora_layers.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
    )
    
    train_dataset = ImageDataset(images_paths_csv=args.images_paths_csv, tokenizer=tokenizer, image_resolution=args.image_resolution, use_cache_emb=False)
    if args.use_cache_emb:
        print('start caching embeddings')
        embeddings, text_masks = [], []
        for i in tqdm(range(len(train_dataset))):
            _, input_ids, attention_mask = train_dataset[i]
            input_ids, attention_mask = input_ids.unsqueeze(0).long().to(accelerator.device), attention_mask.unsqueeze(0).long().to(accelerator.device)
            context = encode(text_encoder, input_ids, attention_mask).to(weight_dtype)
            embeddings.append(context[0].cpu())
            text_masks.append(attention_mask[0].cpu())
        train_dataset.use_cache_emb = True
        train_dataset.set_cache_embeddings(text_masks, embeddings)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    # Prepare everything with our `accelerator`.
    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            first_epoch = 0
    print('global_step =', global_step)
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("training goes brrr")
    uncondition_prob = torch.tensor([1. - args.uncondition_prob, args.uncondition_prob])
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        losses = []
        steps = []
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                if args.use_cache_emb:
                    images, context, attention_mask = batch
                    images, context, attention_mask = images.to(weight_dtype).to(accelerator.device), context.to(weight_dtype).to(accelerator.device), attention_mask.long().to(accelerator.device)

                else:
                    images, input_ids, attention_mask = batch
                    images, input_ids, attention_mask = images.to(weight_dtype).to(accelerator.device), input_ids.long().to(accelerator.device), attention_mask.long().to(accelerator.device)
                
                    context = encode(text_encoder, input_ids, attention_mask).to(weight_dtype)
                uncondition_mask_idx = torch.multinomial(uncondition_prob, images.shape[0], replacement=True).bool()
                context = context * attention_mask.unsqueeze(2)
                context, attention_mask = process_embeds(context, attention_mask)### я хочу это убрать

                
                latents = vae.encode(images).latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                steps += list(torch.clone(timesteps).cpu().numpy()) ###test
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                target = noise
                print(noisy_latents.shape, timesteps.shape, context.shape, attention_mask.shape)
                model_pred = unet(noisy_latents, timesteps, context=context, context_mask=attention_mask,
                                  use_projections=True, return_dict=False, split_context=False, uncondition_mask_idx=uncondition_mask_idx)[0]
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                with torch.no_grad():
                    losses_l = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean(dim=(1, 2, 3))
                    losses_list = list(losses_l.cpu().detach().numpy())
                    losses += losses_list ###


                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                log_step_interval = 50
                logs_keys = [get_key_step(i) for i in range(0, 1000, log_step_interval)]
                logs_dict = {key:[] for key in logs_keys}
                for i in range(len(steps)):
                    logs_dict[get_key_step(steps[i])].append(losses[i])
                filtered_logs_keys = [get_key_step(i) for i in range(0, 1000, log_step_interval) if len(logs_dict[get_key_step(i)]) > 0]
                filtered_logs_dict = {key:float(np.array(logs_dict[key]).mean()) for key in filtered_logs_keys}
                filtered_logs_dict["train_loss"] = train_loss
                #print('type', type(steps[0]), type(losses[0]))
                accelerator.log(filtered_logs_dict, step=global_step)
                train_loss = 0.0
                losses, steps = [], []

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
    accelerator.end_training()


if __name__ == "__main__":
    main()
