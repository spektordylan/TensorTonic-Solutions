import numpy as np

def train_discriminator_step(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    D_W: np.ndarray,
    learning_rate: float,
) -> dict:
    """
    Returns updated discriminator weights and the pre-update loss.
    """
    p_real = sigmoid(real_data @ D_W)
    p_fake = sigmoid(fake_data @ D_W)

    p_real_clipped = np.clip(p_real, 10**-8, 1 - 10**-8)
    p_fake_clipped = np.clip(p_fake, 10**-8, 1 - 10**-8)

    L_D = -np.mean(np.log(p_real_clipped) + np.log(1 - p_fake_clipped))

    grad_L_D = (real_data.T @ (p_real - 1) + fake_data.T @ p_fake)/len(real_data)

    return {'new_discriminator_weights': D_W - learning_rate * grad_L_D,
           'discriminator_loss': L_D}

def sigmoid(x):
    return 1/(1+np.exp(-x))