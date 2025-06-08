import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamPredictor, sam_model_registry
# from lang_sam import LangSAM

# Load the model (choose 'vit_b', 'vit_l', or 'vit_h' for different versions)
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
sam.to(device="cuda" if torch.cuda.is_available() else "cpu")

# Create the predictor
predictor = SamPredictor(sam)
