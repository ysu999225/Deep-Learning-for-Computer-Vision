import random
import unittest

import numpy as np
import torch
import torch.nn as nn
from gradescope_utils.autograder_utils.decorators import weight, visibility, number

from vae import VAE
from score import ScoreNet

SEED = 42


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_error(obj, ref, pred):
    msg = f'{obj} does not match the expected values!\n'
    ae = torch.abs(ref - pred).view(-1)
    msg += f'Total absolute error: {ae.sum().item():.4f}\n'
    msg += f'Mean absolute error: {ae.mean().item():.4f}\n'
    msg += f'Max absolute error: {ae.max().item():.4f}\n'
    return msg


class TestModels(unittest.TestCase):
    def setUp(self):
        pass

    ####################
    # VAE              #
    ####################

    @weight(1.0)
    @number('1.1')
    @visibility('visible')
    def test_vae_reparam(self):
        vae = VAE(nn.Identity(), nn.Identity())
        seed_all(SEED)
        sz = (16, 8)
        mu = torch.randn(*sz)
        logvar = torch.randn(*sz)

        zs = torch.stack([vae.reparameterize(mu, logvar) for _ in range(10000)], dim=0)
        z_sz = tuple(zs.shape[1:])
        self.assertTrue(
            z_sz == sz,
            f'Shape of output from reparameterization is {z_sz}, '
            f'which does not match expected shape {sz}!'
        )

        z_mu = zs.mean(dim=0)
        z_logvar = torch.log(zs.var(dim=0))
        self.assertTrue(
            torch.allclose(z_mu, mu, atol=1e-1),
            log_error('Mean from reparameterization', mu, z_mu)
        )
        self.assertTrue(
            torch.allclose(z_logvar, logvar, atol=1e-1),
            log_error('Log variance from reparameterization', mu, z_mu)
        )

    @weight(3.0)
    @number('1.2')
    @visibility('visible')
    def test_vae_loss(self):
        vae = VAE(lambda x: torch.cat([x, x], dim=-1), nn.Identity())
        seed_all(SEED)
        sz = (100000, 16)
        indata = torch.ones(*sz)

        loss = vae.get_loss(indata)
        # recons loss is E[N(1, e) - 1]^2 = e
        # KL loss is 0.5 * (e + 1 - 1 - 1) * dim = 8 * (e - 1)
        # expect loss is 9e - 8 = 16.46
        gt = torch.FloatTensor([1]).exp() * 9 - 8
        self.assertTrue(
            loss.dim() == 0,
            f'loss should be a scalar, but got shape {loss.shape}!'
        )
        self.assertTrue(
            torch.isclose(loss, gt, atol=1e-2).item(),
            f'loss should be {gt.item():.4f}, but got {loss.item():.4f}!'
        )

    ####################
    # Score Matching   #
    ####################

    @weight(2.0)
    @number('2.1')
    @visibility('visible')
    def test_score_perturb(self):
        noise_level = 10
        scorenet = ScoreNet(nn.Identity(), 10., 1., noise_level, 'linear')
        # seed_all(SEED)
        n_sample = 10000
        x = torch.randn(n_sample, 1000)
        noise, sigma = scorenet.perturb(x)

        # Noise
        self.assertTrue(
            noise.shape == x.shape,
            f'Shape of noise is {noise.shape}, which does not match expected shape {x.shape}!'
        )
        noise_mu = noise.mean()
        self.assertTrue(
            torch.allclose(noise_mu, torch.FloatTensor([0.]), atol=1e-2),
            f'Mean of noise should be 0, but got {noise_mu.item():.4f}!'
        )

        # Sigma
        self.assertTrue(
            tuple(sigma.shape) == (n_sample, 1),
            f'Shape of sigma is {sigma.shape}, which does not match expected shape ({n_sample}, 1)!'
        )
        bincnt = torch.bincount(sigma.squeeze().long())[1:] / n_sample
        self.assertTrue(
            len(bincnt) == noise_level,
            f'Number of bins in sigma distribution is {len(bincnt)}, which does not match expected {noise_level}!'
        )
        self.assertTrue(
            torch.allclose(bincnt, torch.FloatTensor([0.1] * noise_level), atol=1e-1),
            f'Sigma distribution is {bincnt},\n'
            f'which is not uniform!'
        )

        noise_std = noise.std(-1)
        self.assertTrue(
            torch.allclose(noise_std, sigma.squeeze(), atol=1e-1, rtol=1e-1),
            log_error('Standard deviation of noise', sigma[:, 0], noise_std)
        )

    @weight(3.0)
    @number('2.2')
    @visibility('visible')
    def test_score_loss(self):
        noise_level = 10
        scorenet = ScoreNet(nn.Identity(), 1., 0.1, noise_level, 'geometric')
        seed_all(SEED)
        x = torch.zeros(10000, 1000)
        loss = scorenet.get_loss(x)
        self.assertTrue(
            loss.dim() == 0,
            f'loss should be a scalar, but got shape {loss.shape}!'
        )
        gt = ((scorenet.sigmas + 1) ** 2).sum() / 2 / noise_level
        self.assertTrue(
            torch.isclose(loss, gt, rtol=1e-2).item(),
            f'loss should be {gt.item():.4f}, but got {loss.item():.4f}!'
        )

    @weight(2.0)
    @number('2.3')
    @visibility('visible')
    def test_score_sample(self):
        scorenet = ScoreNet(nn.Identity(), 1., 1., 1, 'linear')
        seed_all(SEED)
        n_sample = 10000
        img_size = 256
        x = scorenet.sample(n_sample, img_size, torch.FloatTensor([1.]), 1, 1.)
        self.assertTrue(
            tuple(x.shape) == (1, 1, n_sample, img_size),
            f'Shape of images is {x.shape}, which does not match expected shape (1, 1, {n_sample}, {img_size})!'
        )

        x = x.view(-1)
        mu = x.mean()
        self.assertTrue(
            torch.allclose(mu, torch.FloatTensor([1.]), atol=1e-2),
            f'Mean of images should be 1, but got {mu.item():.4f}!'
        )
        std = x.std()
        std_gt = torch.FloatTensor([2. + 1 / 3]).sqrt()
        self.assertTrue(
            torch.allclose(std, std_gt, atol=1e-2),
            f'Standard deviation of images should be {std_gt.item():.4f}, but got {std.item():.4f}!'
        )


if __name__ == '__main__':
    unittest.main()
