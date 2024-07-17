#%matplotlib widget
#%matplotlib inline
from segment_anything import sam_model_registry
from segment_anything.modeling import PromptEncoder
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional, Tuple
from matplotlib import pyplot as plt
from copy import deepcopy
from torch import nn
from torch.nn import functional as F
import torch
import os
join = os.path.join
import cv2
import numpy as np
import SimpleITK as sitk
import warnings
warnings.filterwarnings('ignore')

import config as CFG

class TextPromptEncoder(PromptEncoder):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int = 1,
        activation = nn.GELU,
        ) -> None:
        super().__init__(embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation)
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        text_encoder.requires_grad_(False)
        self.text_encoder = text_encoder
        self.text_encoder_head = nn.Linear(512, embed_dim)

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        tokens: Optional[torch.Tensor],
    ):
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks, tokens)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if tokens is not None:
            encoder_hidden_states = self.text_encoder(tokens)[0]
            text_embeddings = self.text_encoder_head(encoder_hidden_states)
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings
    
    def _get_batch_size(self, points, boxes, masks, tokens):
        """
        Returns the batch size of the inputs.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        elif tokens is not None:
            return tokens.shape[0]
        else:
            return 1


class MedSAMText(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder,
                device,
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.device = device

    def forward(self, image, tokens):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            tokens=tokens
        )
        low_res_logits, _ = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_logits
    
medsam_ckpt_path =  CFG.medsam_ckpt_path
device = CFG.device
medsam_model = sam_model_registry['vit_b'](checkpoint=medsam_ckpt_path)
medsam_text_demo_checkpoint = CFG.medsam_text_demo_checkpoint
text_prompt_encoder = TextPromptEncoder(
    embed_dim = 256,
    image_embedding_size = (64, 64),
    input_image_size = (1024, 1024),
    mask_in_chans = 1
)
medsam_text_demo = MedSAMText(
    image_encoder=deepcopy(medsam_model.image_encoder),
    mask_decoder=deepcopy(medsam_model.mask_decoder),
    prompt_encoder=text_prompt_encoder,
    device = device
)
medsam_text_demo_weights = torch.load(medsam_text_demo_checkpoint, map_location='cpu')

for key in medsam_text_demo.state_dict().keys():
    if not key.startswith('prompt_encoder.text_encoder.'):
        medsam_text_demo.state_dict()[key].copy_(medsam_text_demo_weights['model'][key])
medsam_text_demo = medsam_text_demo.to(device)
medsam_text_demo.eval()


class TextPromptDemo:
    def __init__(self, model):
        ## Requires transformers only if users would like to try text prompt
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")   
        self.model = model
        self.model.eval()
    
    @torch.no_grad()
    def infer(self, image, text='LUNG NODULE'):
        self.img_size = image.shape[:2]
        if len(image.shape) == 2:
            image = np.repeat(image[:,:,None], 3, -1)
        image_preprocess = self.preprocess_image(image)
        self.image_embeddings = self.model.image_encoder(image_preprocess)

        tokens = self.tokenize_text(text).to(self.model.device)
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points = None,
            boxes = None,
            masks = None,
            tokens = tokens
        )
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=self.image_embeddings, # (B, 256, 64, 64)
            image_pe=self.model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
            )
        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
        low_res_pred = F.interpolate(
            low_res_pred,
            size = (self.img_size[0], self.img_size[1]),
            mode = 'bilinear',
            align_corners = False
        )
        low_res_pred = low_res_pred.detach().cpu().numpy().squeeze()
        seg = np.uint8(low_res_pred > 0.5)

        return seg

    def preprocess_image(self, image):
        img_resize = cv2.resize(
            image,
            (1024, 1024),
            interpolation=cv2.INTER_CUBIC
        )
        # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
        # convert the shape to (3, H, W)
        assert np.max(img_resize)<=1.0 and np.min(img_resize)>=0.0, 'image should be normalized to [0, 1]'
        img_tensor = torch.tensor(img_resize).float().permute(2, 0, 1).unsqueeze(0).to(self.model.device)

        return img_tensor
    
    def tokenize_text(self, text):
        """
        Tokenize text using CLIP tokenizer
        """
        return self.tokenizer(
            text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt" 
        ).input_ids.squeeze(1)

text_prompt_demo = TextPromptDemo(medsam_text_demo)