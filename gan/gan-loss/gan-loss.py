import numpy as np

def gan_losses(real_probs: np.ndarray, fake_probs: np.ndarray) -> dict:
    """
    Returns discriminator_loss and generator_loss as Python floats.
    """
    clipped_real_probs = np.clip(real_probs, 10**-8, 1 - 10**-8)
    clipped_fake_probs = np.clip(fake_probs, 10**-8, 1 - 10**-8)

    N = len(real_probs)
    
    assert N == len(fake_probs)

    return {"discriminator_loss": -np.mean(np.log(clipped_real_probs) + np.log(1-clipped_fake_probs)),
            "generator_loss": -np.mean(np.log(clipped_fake_probs))}