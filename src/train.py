import einops as eo
import torch
import matplotlib.pyplot as plt
from src.dataset import Spiral2D, SimpleNormal2D
from src.utils import show_plot, bb, save_gif, read_plot
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm


def self_outer(x):
    return torch.einsum('... c, ... d -> ... c d', x, x)

def gmm_plot(pts, dists, pi, ax=None):
    mean = dists.mean.detach().cpu()
    pi = pi.detach().cpu()
    pts = pts.detach().cpu()
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
    ax.scatter(pts[:, 0], pts[:, 1], s=0.3)
    ax.scatter(mean[:, 0], mean[:, 1], c="red", s=80)

    pi_cumsum = torch.cumsum(pi, dim=-1)
    n = 5000
    sample_pts_idx = (torch.rand(n)[:, None] >= pi_cumsum[None, :]).sum(dim=-1).type(torch.int)
    sampled_pts = dists.sample((n,)).detach().cpu()
    sampled_pts_selected = sampled_pts[torch.arange(n), sample_pts_idx, :]
    ax.scatter(sampled_pts_selected[:, 0], sampled_pts_selected[:, 1], s=0.4, c="green")

def gmm(pts, n_components=3, iterations=100):
    b, d = pts.shape
    device = pts.device
    dists = torch.distributions.MultivariateNormal(torch.randn(n_components, d, device=device), torch.eye(d, device=device).repeat(n_components, 1, 1))
    pi = torch.ones(n_components, device=device) / n_components
    frames = []
    for i in tqdm(range(iterations)):
        # b c
        my_pts = eo.rearrange(pts, "b d -> b d")
        my_means = eo.rearrange(dists.mean, "c d -> c d")
        my_pi = eo.rearrange(pi, "c -> c")

        logits = dists.log_prob(my_pts[:, None, :])
        logits = logits + torch.log(my_pi[None, :])
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        # logits are logp(c | x)
        ps = torch.softmax(logits, dim=-1)

        normalization = ps.sum(dim=0) # c
        new_mus = eo.einsum(pts, ps, "b d, b c -> c d") / normalization[:, None]
        new_variances = eo.einsum(
            self_outer(pts[:, None, :] - my_means[None, :, :]),
            ps,
            "b c d1 d2, b c -> c d1 d2"
        ) / normalization[:, None, None]
        new_pis = eo.reduce(ps, "b c -> c", "sum")
        new_pis = new_pis / new_pis.sum()

        dists = torch.distributions.MultivariateNormal(new_mus, new_variances + 1e-6)
        pi = new_pis

        gmm_plot(pts, dists, pi)
        show_plot("gmm")
        frames.append(read_plot("gmm"))
    save_gif(frames, name="gmm")
    return dists, pi


def variational_autoencoder(pts, iterations=20):
    b, d = pts.shape
    device = pts.device
    q_net = nn.Sequential(
        nn.Linear(d, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 2*d),  # d for mean, d for log_std
    ).to(device)
    p_net = nn.Sequential(
        nn.Linear(d, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 2*d),  # d for mean, d for log_std
    ).to(device)
    optimizer = torch.optim.Adam([*q_net.parameters(), *p_net.parameters()], lr=1e-4)

    prior_dist = torch.distributions.Normal(torch.zeros(d, device=device), torch.ones(d, device=device))

    def conditional_p_dist(z):
        mu_sigma = p_net(z)
        mu = mu_sigma[:, :d]
        log_sigma = mu_sigma[:, d:]  # Allow different sigma for each dimension
        sigma = 0.05 + torch.exp(log_sigma)
        return torch.distributions.Normal(mu, sigma)

    def conditional_q_dist(x):
        mu_sigma = q_net(x)
        mu = mu_sigma[:, :d]
        log_sigma = mu_sigma[:, d:]  # Allow different sigma for each dimension
        sigma = 0.05 + torch.exp(log_sigma)
        return torch.distributions.Normal(mu, sigma)
    
    frames = []
    losses = []
    loader = DataLoader(pts, batch_size=256, shuffle=True)
    for iter in tqdm(range(iterations)):
        """
        we need to optimize the variational loss
        L(x) = -E[log p(x | z)] + KL(q(z | x) || p(z))
        """
        for x in tqdm(loader):
            q_x_dist = conditional_q_dist(x)
            
            # Sample multiple z for better estimation (using reparameterization trick)
            z = q_x_dist.rsample()  # Use rsample() instead of sample() for reparameterization
            p_z_dist = conditional_p_dist(z)
            
            # Current implementation:
            # L0 = -p_z_dist.log_prob(x).mean()  # This is incorrect
            # L1 = torch.distributions.kl_divergence(q_x_dist, prior_dist).mean()
            
            # Correct ELBO implementation:
            reconstruction_loss = -p_z_dist.log_prob(x).sum(dim=-1).mean()
            kl_loss = torch.distributions.kl_divergence(q_x_dist, prior_dist).mean()
            beta = 1.0  # Can be adjusted
            loss = reconstruction_loss + beta * kl_loss
            
            # For debugging, you might want to track both terms separately
            losses.append({
                'total': loss.item(),
                'reconstruction': reconstruction_loss.item(),
                'kl': kl_loss.item()
            })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        plt.figure(figsize=(10, 10))
        plt.subplot(3, 1, 1)
        plt.plot([l['total'] for l in losses])
        plt.title('Total Loss')
        plt.subplot(3, 1, 2)
        plt.plot([l['reconstruction'] for l in losses])
        plt.title('Reconstruction Loss')
        plt.subplot(3, 1, 3)
        plt.plot([l['kl'] for l in losses])
        plt.title('KL Loss')
        show_plot(f"vae_loss")

        nz = 100
        z = prior_dist.sample((nz,))
        p_z_dist = conditional_p_dist(z)
        gmm_plot(pts, p_z_dist, torch.ones(nz) / nz)
        show_plot("vae_reconstruction")
        frames.append(read_plot(f"vae_reconstruction"))

    save_gif(frames, name="vae_reconstruction")
    return q_net, p_net, prior_dist


def train_vae_main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = Spiral2D(n=3000)
    # dataset = SimpleDataset2D(each=2000, std=0.1)
    # dataset = SimpleNormal2D(n=2000, std=1.0)

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    pts = next(iter(loader)).to(device)
    q_net, p_net, prior_dist = variational_autoencoder(pts, iterations=50)


def train_gmm_main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = Spiral2D(n=3000)
    # dataset = SimpleDataset2D(each=2000, std=0.1)
    # dataset = SimpleNormal2D(n=2000, std=1.0)

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    pts = next(iter(loader)).to(device)
    dists, pi = gmm(pts, n_components=40, iterations=100)
    gmm_plot(pts, dists, pi)
    show_plot("gmm")
