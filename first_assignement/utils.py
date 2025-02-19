import torch
import torch.nn as nn
from scipy.linalg import sqrtm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gradient_penalty(D, x, y):
    # Calculate interpolation
    b = x.shape[0]
    n = y.shape[0]
    alpha = torch.rand((b, n, 1), device=device)
    interp = (alpha * y[None, :, :] + (1 - alpha) * x[:, None, :]).flatten(end_dim=1)
    interp.requires_grad_()

    # Calculate probability of interpolated examples
    Di = D(interp).view(-1)

    # Calculate gradients of probabilities with respect to examples
    gradout = torch.ones(Di.size()).to(device)
    gradients = torch.autograd.grad(
        outputs=Di,
        inputs=interp,
        grad_outputs=gradout,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)

    # Return gradient penalty
    return ((gradients_norm - 1) ** 2).mean()


#     return ((gradients_norm - 1) ** 2*(gradients_norm>1)).mean()


def lipconstant(D, x, y):
    # Calculate interpolation
    b = x.shape[0]
    n = y.shape[0]
    alpha = torch.rand((b, n, 1), device=device)
    interp = (alpha * y[None, :, :] + (1 - alpha) * x[:, None, :]).flatten(end_dim=1)
    interp.requires_grad_()

    # Calculate probability of interpolated examples
    Di = D(interp).view(-1)

    # Calculate gradients of probabilities with respect to examples
    gradout = torch.ones(Di.size()).to(device)
    gradients = torch.autograd.grad(
        outputs=Di,
        inputs=interp,
        grad_outputs=gradout,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1))

    # Return gradient penalty
    return torch.mean(gradients_norm)


def vae_loss(reconstructed_x, x, mu, logvar, beta_kl=0.1):
    recon_loss = nn.BCEWithLogitsLoss()(reconstructed_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta_kl * kl_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def M_lipconstant(D, x, y):
    b = x.shape[0]
    n = y.shape[0]
    alpha = torch.rand((b, n, 1, 1, 1), device=x.device)
    interp = (alpha * y.unsqueeze(1) + (1 - alpha) * x.unsqueeze(1)).view(
        -1, *x.shape[1:]
    )
    interp.requires_grad_()

    Di = D(interp).view(-1)

    grad_outputs = torch.ones_like(Di, device=x.device)
    gradients = torch.autograd.grad(
        outputs=Di,
        inputs=interp,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients_norm = torch.norm(gradients.view(gradients.shape[0], -1), p=2, dim=1)

    return gradients_norm.mean()


def M_gradient_penalty(D, x, y):
    b = x.shape[0]
    n = y.shape[0]
    alpha = torch.rand((b, n, 1, 1, 1), device=x.device)  # Shape for broadcasting
    interp = (alpha * y.unsqueeze(1) + (1 - alpha) * x.unsqueeze(1)).view(
        -1, *x.shape[1:]
    )
    interp.requires_grad_()

    Di = D(interp).view(-1)

    grad_outputs = torch.ones_like(Di, device=x.device)
    gradients = torch.autograd.grad(
        outputs=Di,
        inputs=interp,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients_norm = torch.norm(gradients.view(gradients.shape[0], -1), p=2, dim=1)

    return ((gradients_norm - 1) ** 2).mean()


# QUESTION 4 ############


def compute_statistics(features):
    mu = torch.mean(features, dim=0)
    sigma = torch.cov(features.T)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    diff_sq = torch.dot(diff, diff)

    covmean = torch.tensor(sqrtm(sigma1 @ sigma2).real, dtype=torch.float32)

    return diff_sq + torch.trace(sigma1 + sigma2 - 2 * covmean)


def compute_mfd(real_images, fake_images, model):
    real_features = extract_features(real_images, model)
    fake_features = extract_features(fake_images, model)

    mu_real, sigma_real = compute_statistics(real_features)
    mu_fake, sigma_fake = compute_statistics(fake_features)

    mfd_score = frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return mfd_score.item()


def extract_features(images, model):
    with torch.no_grad():
        images = F.interpolate(images, size=(28, 28))
        features = model(images)
        return features.view(features.size(0), -1)
