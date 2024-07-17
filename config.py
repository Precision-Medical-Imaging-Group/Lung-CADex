import torch

seed = 1008
debug = True
batch_size = 8
num_workers = 0
lr = 1e-3
weight_decay = 1e-3
patience = 2
factor = 0.5
epochs = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best.pt"

model_name = 'resnet50'
image_embedding = 2048
text_embedding = 8
text_encoder_model = "distilbert-base-uncased"
text_tokenizer = "distilbert-base-uncased"
max_length = 200

pretrained = True # for both image encoder and text encoder
trainable = True # for both image encoder and text encoder
temperature = 1.0

# image size
size = 96

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 32 
dropout = 0.1

medsam_ckpt_path = "C:\\Users\\fu057938\\GeoCLIP_Project\\Trained Model\\medsam_vit_b.pth"
medsam_text_demo_checkpoint ="C:\\Users\\fu057938\\GeoCLIP_Project\\Trained Model\\With text prompt\\text_prompt_workdir_1\\medsam_text_prompt_best.pth"