import torch
import torch.nn as nn
import numpy as np


class ScoreNet(nn.Module):
    """Score matching model"""

    def __init__(self, scorenet, sigma_begin, sigma_end, noise_level, sigma_type='geometric'):
        """
        :param scorenet: an `nn.Module` instance that computes the score of the input images
        :param sigma_begin: the largest sigma value
        :param sigma_end: the smallest sigma value
        :param noise_level: the number of noise levels
        :param sigma_type: the type of sigma distribution, 'geometric' or 'linear'
        """
        super().__init__()
        self.scorenet = scorenet

        self.sigmas: torch.Tensor
        sigmas = self.get_sigmas(sigma_begin, sigma_end, noise_level, sigma_type)
        self.register_buffer('sigmas', sigmas)  # (num_noise_level,)

    @staticmethod
    def get_sigmas(sigma_begin, sigma_end, noise_level, sigma_type='geometric'):
        """
        Get the sigmas used to perturb the images
        :param sigma_begin: the largest sigma value
        :param sigma_end: the smallest sigma value
        :param noise_level: the number of noise levels
        :param sigma_type: the type of sigma distribution, 'geometric' or 'linear'
        :return: sigmas of shape (num_noise_level,)
        """
        if sigma_type == 'geometric':
            sigmas = torch.FloatTensor(np.geomspace(
                sigma_begin, sigma_end,
                noise_level
            ))
        elif sigma_type == 'linear':
            sigmas = torch.FloatTensor(np.linspace(
                sigma_begin, sigma_end, noise_level
            ))
        else:
            raise NotImplementedError(f'sigma distribution {sigma_type} not supported')
        return sigmas

    def perturb(self, batch):
        """
        Perturb images with Gaussian noise.
        You should randomly choose a sigma from `self.sigmas` for each image in the batch.
        Use that sigma as the standard deviation of the Gaussian noise added to the image.
        :param batch: batch of images of shape (N, D)
        :return: noises added to images (N, D)
                 sigmas used to perturb the images (N, 1)
        """
        batch_size = batch.size(0)
        device = batch.device
        # TODO: Implement the perturb
        # Below is the placeholder code you should modify
        used_sigmas = torch.ones(batch_size, dtype=torch.float, device=device)
        # randomly choose a sigma from `self.sigmas` for each image in the batch.
        random = torch.randint(len(self.sigmas),(batch_size,))
        used_sigmas = self.sigmas[random]
        # Use that sigma as the standard deviation of the Gaussian noise added to the image.
        # initialize and fill in values, added to the image
        noise = torch.zeros_like(batch) 
        noise = torch.randn_like(batch) *  used_sigmas.unsqueeze(1)
        return noise, used_sigmas.unsqueeze(1)






    @torch.no_grad()
    def sample(self, batch_size, img_size, sigmas=None, n_steps_each=10, step_lr=2e-5):
        """
        Run Langevin dynamics to generate images
        :param batch_size: batch size of the images
        :param img_size: image size of the images of D = H * W
        :param sigmas: sequence of sigmas used to run the annealed Langevin dynamics
        :param n_steps_each: number of steps for each sigma
        :param step_lr: initial step size
        :return: image trajectories (num_sigma, num_step, N, D)
        """

        self.eval()
        if sigmas is None:
            sigmas = self.sigmas

        # In NCSNv2, the initial x is sampled from a uniform distribution instead of a Gaussian distribution
        x = torch.rand(batch_size, img_size, device=sigmas.device)

        traj = []
        for sigma in sigmas:
            # scale the step size according to the smallest sigma
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            # run Langevin dynamics
            for step in range(n_steps_each):
                score = self.get_score(x, sigma)
                # TODO: Implement the Langevin dynamics
                # Append the new trajectory to `traj`
                noise = torch.randn_like(x)
                x += step_size * score + torch.sqrt(2 * step_size) * noise
                traj.append(x.clone())
        traj = torch.stack(traj, dim=0).view(sigmas.size(0), n_steps_each, *x.size())
        return traj


    def get_score(self, x, sigma):
        """
        Calculate the score of the input images
        :param x: images of (N, D)
        :param sigma: the sigma used to perturb the images, either a float or a tensor of shape (N, 1)
        :return: the score of the input images, of shape (N, D)
        """
        # In NCSNv2, the score is divided by sigma (i.e., noise-conditioned)
        out = self.scorenet(x) / sigma
        return out

    def get_loss(self, x):
        """
        Calculate the score loss.
        The loss should be averaged over the batch dimension and the image dimension.
        :param x: images of (N, D)
        :return: score loss, a scalar tensor
        """
        # TODO: Implement this function
        # call the perturb function
        noise, used_sigmas = self.perturb(x)
        perturb_x = x + noise
        score = self.get_score(perturb_x, used_sigmas)
        ## The score loss in NCSNv2
        score_loss = (score - (x - perturb_x) / used_sigmas ** 2) ** 2 * used_sigmas ** 2
        #1/2
        score_loss = 0.5 * score_loss.mean()  
        return score_loss

        
    def forward(self, x):
        """
        Calculate the result of the score net (not noise-conditioned)
        :param x: images of (N, D)
        :return: the result of the score net, of shape (N, D)
        """
        return self.scorenet(x)
