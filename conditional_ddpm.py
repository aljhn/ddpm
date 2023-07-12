import random
import math

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST

seed = 42069
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def show_images(images, title=""):
    images = images.detach().cpu().numpy()
    fig = plt.figure(figsize=(4, 4))
    cols = math.ceil(len(images) ** (1 / 2))
    rows = math.ceil(len(images) / cols)
    for r in range(rows):
        for c in range(cols):
            idx = cols * r + c
            ax = fig.add_subplot(rows, cols, idx + 1)
            ax.axis("off")
            if idx < len(images):
                ax.imshow(images[idx][0], cmap="gray")
    fig.suptitle(title, fontsize=18)
    plt.show()


class ConditionalDDPM:
    """
    Conditional Denoising Diffusion Probabilistic Model, based on Classifier-Free Diffusion Guidance.
    Class contains methods for sampling from a specific class and training on a dataset.
    """

    def __init__(
        self,
        model,
        image_size,
        steps,
        beta_min,
        beta_max,
        w,
        p_unconditional,
    ):
        self.model = model
        self.image_size = image_size
        self.steps = steps
        self.beta = torch.linspace(beta_min, beta_max, steps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma = self.beta**0.5
        self.w = w
        self.p_unconditional = p_unconditional

    def p_sample(self, x_t, t, z, epsilon):
        """
        Sample from the reverse process. Removes noise from the processed image.
        """
        return (
            1.0
            / ((self.alpha[t]) ** 0.5)
            * (x_t - (1 - self.alpha[t]) / ((1 - self.alpha_bar[t]) ** 0.5) * epsilon)
            + self.sigma[t] * z
        )

    def sample(self, batch_size, c):
        """
        Sample a batch of images from the given class using the diffusion model.
        The class to sample is put into a tensor before being inputed into the network.
        It could be possible to sample different classes at the same time, but that is not implemented here.
        As the network requires two forward passes, the input variables x, t and c are concatenated with themselves
        so that the two forward passes are done as one batched forward pass.
        The conditionals are a tensor consisting of ones for the guided pass and zeros for the unguided pass.
        """
        c = torch.ones((batch_size,), dtype=torch.int64, device=device) * c
        c_input = torch.cat((c, c), dim=0)
        conditionals = torch.ones((2 * batch_size, 1), device=device)
        conditionals[batch_size:, 0] = 0.0
        x_t = torch.randn(
            (batch_size, 1, self.image_size, self.image_size), device=device
        )
        for t in tqdm(range(self.steps - 1, -1, -1)):
            if t > 0:
                z = torch.randn(x_t.shape, device=device)
            else:  # Don't add any noise on the last iteration
                z = torch.zeros(x_t.shape, device=device)
            t_input = torch.ones((2 * batch_size, 1), device=device) * t
            x_t_input = torch.cat((x_t, x_t), dim=0)
            with torch.no_grad():
                epsilon_pred = self.model(x_t_input, t_input, c_input, conditionals)
            epsilon_pred_guided = epsilon_pred[:batch_size, :, :, :]
            epsilon_pred_unguided = epsilon_pred[batch_size:, :, :, :]
            epsilon_tilde_t = (
                1.0 + self.w
            ) * epsilon_pred_guided - self.w * epsilon_pred_unguided
            x_t = torch.vmap(self.p_sample, in_dims=(0, None, 0, 0))(
                x_t, t, z, epsilon_tilde_t
            )
        return x_t

    def q_sample(self, x_0, t, epsilon):
        """
        Sample from the forward process. Adds gaussian noise to an initial data point.
        """
        return (self.alpha_bar[t] ** 0.5) * x_0 + (
            (1 - self.alpha_bar[t]) ** 0.5
        ) * epsilon

    def train(self, x_0, c):
        """
        Performs one training iteration for the diffusion model.
        Takes in a batch of images along with class labels to train on.
        """
        batch_size = x_0.shape[0]
        t = torch.randint(1, self.steps, size=(batch_size, 1), device=device)
        epsilon = torch.randn(
            (batch_size, 1, self.image_size, self.image_size), device=device
        )
        x_input = torch.vmap(self.q_sample)(x_0, t, epsilon)
        conditionals = torch.rand((batch_size, 1), device=device) > self.p_unconditional
        epsilon_pred = self.model(x_input, t, c, conditionals)
        loss = torch.sum((epsilon - epsilon_pred) ** 2, dim=1)
        loss = torch.mean(loss)
        return loss


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding from the attention paper.
    Used for embedding the time input to the network.
    """

    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, t):
        indexes = torch.arange(0, self.dimension, 2, device=device)
        sin_embedding = torch.sin(t / (10000.0 ** (indexes / self.dimension)))
        cos_embedding = torch.cos(t / (10000.0 ** (indexes / self.dimension)))
        embedding = torch.stack((sin_embedding, cos_embedding), dim=1)
        embedding = embedding.transpose(1, 2).reshape(t.shape[0], self.dimension)
        return embedding


class SelfAttentionBlock(nn.Module):
    """
    Multi-Headed Self Attention modified with 2D convolutions instead of linear layers.
    """

    def __init__(self, dimension, heads, hidden_dimension, groups):
        super().__init__()
        self.dimension = dimension
        self.heads = heads
        self.hidden_dimension = hidden_dimension

        self.layer_q = nn.Conv2d(
            dimension, heads * hidden_dimension, kernel_size=1, stride=1, padding=0
        )
        self.layer_k = nn.Conv2d(
            dimension, heads * hidden_dimension, kernel_size=1, stride=1, padding=0
        )
        self.layer_v = nn.Conv2d(
            dimension, heads * hidden_dimension, kernel_size=1, stride=1, padding=0
        )
        self.layer_out = nn.Conv2d(
            heads * hidden_dimension, dimension, kernel_size=1, stride=1, padding=0
        )
        self.softmax = nn.Softmax(dim=-1)
        self.normalization = nn.GroupNorm(num_groups=groups, num_channels=dimension)

    def forward(self, x):
        batch_size = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]

        x = self.normalization(x)

        q = self.layer_q(x)
        k = self.layer_k(x)
        v = self.layer_v(x)

        # Permute and reshape he tensors so that the matrix multiplications work out correctly.
        # For q and v, the channel dimension is moved last.
        # All three tensors are also flattened along the height and width dimensions, thus making each of them a batch of matrices.
        q = q.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, self.heads * self.hidden_dimension
        )
        k = k.reshape(batch_size, self.heads * self.hidden_dimension, height * width)
        v = v.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, self.heads * self.hidden_dimension
        )

        # Batched matrix multiplication. The channel dimensions for both q and k are multiplied together.
        # This means that the attention scores are comparing the similarity of pixel positions, instead of channel positions.
        qk = torch.bmm(q, k)
        qk = self.softmax(qk / (self.hidden_dimension**0.5))
        attention = torch.bmm(qk, v)
        # Move the channel dimension back to the second dimension as is the convention.

        attention = attention.reshape(
            batch_size, height, width, self.heads * self.hidden_dimension
        ).permute(0, 3, 1, 2)

        out = self.layer_out(attention)  # Do one final layer after the self attention.
        return x + out


class ResidualBlock(nn.Module):
    """
    Fully convolutional residual block using the SiLU activation function and group normalization layers, inspired by the original paper.
    The time embedding t and class label c is sent through linear layers to make the shape fit with the input image x.
    """

    def __init__(
        self,
        channels,
        groups,
        time_embedding_dimension,
        class_embedding_dimension,
        image_size,
    ):
        super().__init__()
        self.layer1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.layer_t1 = nn.Linear(time_embedding_dimension, time_embedding_dimension)
        self.layer_t2 = nn.Linear(
            time_embedding_dimension, channels * image_size * image_size
        )
        self.layer_c1 = nn.Linear(time_embedding_dimension, time_embedding_dimension)
        self.layer_c2 = nn.Linear(
            class_embedding_dimension, channels * image_size * image_size
        )
        self.activation = nn.SiLU()
        self.normalization1 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.normalization2 = nn.GroupNorm(num_groups=groups, num_channels=channels)

    def forward(self, x, t, c):
        x_init = x

        x = self.normalization1(x)
        x = self.layer1(x)
        x = self.activation(x)

        t = self.layer_t1(t)
        t = self.activation(t)
        t = self.layer_t2(t)
        t = t.reshape(x.shape)
        x += t

        c = self.layer_c1(c)
        c = self.activation(c)
        c = self.layer_c2(c)
        c = c.reshape(x.shape)
        x += c

        x = self.normalization2(x)
        x = self.layer2(x)
        x = self.activation(x)

        return x + x_init


class UNet(nn.Module):
    """
    Network used for learning the reverse diffusion process.
    Consists of a U-Net style architecture based on fully convolutional residual blocks together with a series of downsampling and upsampling operations.
    The same number of downsamples and upsamples are used, and the state at each step during the downsampling is concatenated to the corresponding step in the upsampling.
    The time t is send through a sinusoidal positional embedding followed by some linear layers, and added as an input to each residual block.
    The class labels c are transformed to a one-hot encoding, and then multiplied with the conditionals. The conditionals are either a 1 or a 0, which will disable the classes when the process is unguided.
    Multi-Headed Self Attention blocks working on the image state is added at the middle positions of each stage.
    """

    def __init__(
        self,
        hidden_channels,
        groups,
        attention_heads,
        time_embedding_dimension,
        class_embedding_dimension,
        num_classes,
    ):
        super().__init__()
        self.positional_embedding = PositionalEmbedding(time_embedding_dimension)
        self.layer_t1 = nn.Linear(time_embedding_dimension, time_embedding_dimension)
        self.layer_t2 = nn.Linear(time_embedding_dimension, time_embedding_dimension)
        self.layer_c1 = nn.Linear(num_classes, class_embedding_dimension)
        self.layer_c2 = nn.Linear(class_embedding_dimension, class_embedding_dimension)
        self.num_classes = num_classes

        self.layer_down1 = nn.Conv2d(
            1, hidden_channels, kernel_size=3, stride=1, padding=1
        )
        self.layer_down2 = ResidualBlock(
            hidden_channels,
            groups,
            time_embedding_dimension,
            class_embedding_dimension,
            image_size,
        )
        self.layer_down3 = ResidualBlock(
            hidden_channels,
            groups,
            time_embedding_dimension,
            class_embedding_dimension,
            image_size // 2,
        )
        self.layer_down4 = SelfAttentionBlock(
            hidden_channels, attention_heads, hidden_channels, groups
        )
        self.layer_down5 = ResidualBlock(
            hidden_channels,
            groups,
            time_embedding_dimension,
            class_embedding_dimension,
            image_size // 4,
        )

        self.layer_middle1 = ResidualBlock(
            hidden_channels,
            groups,
            time_embedding_dimension,
            class_embedding_dimension,
            image_size // 4,
        )
        self.layer_middle2 = SelfAttentionBlock(
            hidden_channels, attention_heads, hidden_channels, groups
        )
        self.layer_middle3 = ResidualBlock(
            hidden_channels,
            groups,
            time_embedding_dimension,
            class_embedding_dimension,
            image_size // 4,
        )

        self.layer_up1 = ResidualBlock(
            2 * hidden_channels,
            groups,
            time_embedding_dimension,
            class_embedding_dimension,
            image_size // 4,
        )
        self.layer_up2 = ResidualBlock(
            3 * hidden_channels,
            groups,
            time_embedding_dimension,
            class_embedding_dimension,
            image_size // 2,
        )
        self.layer_up3 = SelfAttentionBlock(
            3 * hidden_channels, attention_heads, 3 * hidden_channels, groups
        )
        self.layer_up4 = ResidualBlock(
            4 * hidden_channels,
            groups,
            time_embedding_dimension,
            class_embedding_dimension,
            image_size,
        )
        self.layer_up5 = nn.Conv2d(
            4 * hidden_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.activation = nn.SiLU()

    def forward(self, x, t, c, conditionals=1.0):
        t_embed = self.positional_embedding(t)
        t_embed = self.layer_t1(t_embed)
        t_embed = self.activation(t_embed)
        t_embed = self.layer_t2(t_embed)

        c = nn.functional.one_hot(c, num_classes=self.num_classes).type(torch.float32)
        c *= conditionals
        c_embed = self.layer_c1(c)
        c_embed = self.activation(c_embed)
        c_embed = self.layer_c2(c_embed)

        x1 = self.layer_down1(x)
        x1 = self.activation(x1)
        x1 = self.layer_down2(x1, t_embed, c_embed)
        x1 = self.activation(x1)

        x2 = self.downsample(x1)
        x2 = self.layer_down3(x2, t_embed, c_embed)
        x2 = self.activation(x2)
        x2 = self.layer_down4(x2)
        x2 = self.activation(x2)

        x3 = self.downsample(x2)
        x3 = self.layer_down5(x3, t_embed, c_embed)
        x3 = self.activation(x3)

        xx = self.layer_middle1(x3, t_embed, c_embed)
        xx = self.activation(xx)
        xx = self.layer_middle2(xx)
        xx = self.activation(xx)
        xx = self.layer_middle3(xx, t_embed, c_embed)
        xx = self.activation(xx)

        xx = torch.cat((xx, x3), dim=1)
        xx = self.layer_up1(xx, t_embed, c_embed)
        xx = self.activation(xx)

        xx = self.upsample(xx)
        xx = torch.cat((xx, x2), dim=1)
        xx = self.layer_up2(xx, t_embed, c_embed)
        xx = self.activation(xx)
        xx = self.layer_up3(xx)
        xx = self.activation(xx)

        xx = self.upsample(xx)
        xx = torch.cat((xx, x1), dim=1)
        xx = self.layer_up4(xx, t_embed, c_embed)
        xx = self.activation(xx)
        xx = self.layer_up5(xx)

        return xx


if __name__ == "__main__":
    batch_size = 64
    image_size = 28
    steps = 1000
    beta_min = 1e-4
    beta_max = 0.02

    epochs = 10
    lr = 1e-4

    hidden_channels = 200
    groups = 10
    attention_heads = 5
    time_embedding_dimension = 100
    class_embedding_dimension = 100
    num_classes = 10

    w = 0.3
    p_unconditional = 0.1

    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])

    dataset = MNIST("./datasets", download=True, train=True, transform=transform)
    loader = DataLoader(dataset, batch_size, shuffle=True)

    model = UNet(
        hidden_channels,
        groups,
        attention_heads,
        time_embedding_dimension,
        class_embedding_dimension,
        num_classes,
    )
    model.to(device)

    ddpm = ConditionalDDPM(
        model, image_size, steps, beta_min, beta_max, w, p_unconditional
    )

    # samples = ddpm.sample(16, 0)
    # show_images(samples)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    losses = []

    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        for iteration, (x_batch, y_batch) in enumerate(loader):
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            loss = ddpm.train(x_batch, y_batch)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(
                {
                    "Loss": loss.item(),
                    "Iteration": f"{iteration} / {60000 // batch_size}",
                }
            )
            losses.append(loss.item())
        scheduler.step()

    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.show()

    for i in range(num_classes):
        samples = ddpm.sample(16, i)
        show_images(samples)
