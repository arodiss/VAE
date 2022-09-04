from vae import VAE
import torch

BEST_MODEL_CKPT = "best_model/checkpoints/last.ckpt"


def get_best_model():
    model = VAE(latent_dim=4, interim_dim=8)
    state_dict = torch.load(BEST_MODEL_CKPT)['state_dict']
    state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model
