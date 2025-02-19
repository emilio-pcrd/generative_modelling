import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(torch.nn.Module):
    def __init__(self, n_in, n_out, n_hid=10, nlayers=3, device="cuda"):
        super(Generator, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_hid = n_hid
        self.nlayers = nlayers
        self.hidden = nn.ModuleList()

        for n in range(nlayers):
            n_in_t = n_in if n == 0 else n_hid
            self.hidden.append(
                nn.Sequential(nn.Linear(n_in_t, n_hid), nn.ELU(1)).to(device)
            )

        self.out = nn.Sequential(nn.Linear(n_hid, n_out), nn.Sigmoid()).to(device)

        self.apply(self._init_weights)

    def forward(self, x):
        for n in range(self.nlayers):
            x = self.hidden[n](x)
        x = self.out(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, 1.0)
            if module.bias is not None:
                module.bias.data.zero_()


class DGAN(nn.Module):
    def __init__(self, n_in, n_hid=10):
        super(DGAN, self).__init__()

        self.n_hid = n_hid
        self.n_in = n_in

        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, 1)

    def forward(self, x):
        y = nn.LeakyReLU(negative_slope=0.2)(self.fc1(x))
        y = nn.LeakyReLU(negative_slope=0.2)(self.fc2(y))
        y = nn.Sigmoid()(self.fc3(y))
        return y


class DWGAN(nn.Module):
    def __init__(self, n_in, n_hid=10):
        super(DWGAN, self).__init__()

        self.n_hid = n_hid
        self.n_in = n_in

        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, 1)

    def forward(self, x):
        y = nn.LeakyReLU(negative_slope=0.2)(self.fc1(x))
        y = nn.LeakyReLU(negative_slope=0.2)(self.fc2(y))
        y = self.fc3(y)
        return y


class VAE(nn.Module):
    def __init__(self, n_in, latent_dim, n_hid):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_in, n_hid), nn.ReLU(), nn.Linear(n_hid, n_hid), nn.ReLU()
        )

        self.mu = nn.Linear(n_hid, latent_dim)
        self.logvar = nn.Linear(n_hid, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_in),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


####################
#   QUESTION 2     #
####################


# Size  of generator input
nz = 100
# Size of feature maps in generator and discriminator
ngf, ndf = 64, 64


class MGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=nz,
                out_channels=ngf * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(
                in_channels=ngf * 8,
                out_channels=ngf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(
                in_channels=ngf * 4,
                out_channels=ngf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(
                in_channels=ngf * 2,
                out_channels=ngf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(
                in_channels=ngf,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=2,
                bias=False,
            ),
            nn.Tanh(),
            # output size. 1 x 28 x 28
        )

    def forward(self, input):
        return self.main(input)

    def _init_weights(self, module):
        if (
            isinstance(module, nn.Linear)
            or isinstance(module, nn.ConvTranspose2d)
            or isinstance(module, nn.Conv2d)
        ):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()


class MDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is 1 x 28 x 28
            nn.Conv2d(
                in_channels=1,
                out_channels=ndf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 15 x 15
            nn.Conv2d(
                in_channels=ndf,
                out_channels=ndf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(
                in_channels=ndf * 2,
                out_channels=ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 5 x 5
            nn.Conv2d(
                in_channels=ndf * 4,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


class MnistVariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim=200, z_dim=20):
        super().__init__()

        # encoder
        self.img2hid = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )
        # gaussian latent space -> mu, std
        self.hid2mu = nn.Linear(hid_dim, z_dim)
        self.hid2std = nn.Linear(hid_dim, z_dim)

        # decoder
        self.z2hid = nn.Sequential(
            nn.Linear(z_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )
        self.hid2img = nn.Linear(hid_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        # q_phi(z|x)
        h = self.relu(self.img2hid(x))
        mu = self.hid2mu(h)
        sigma = self.hid2std(h)

        return mu, sigma

    def decode(self, z):
        # p_theta(x|z)
        h = self.relu(self.z2hid(z))
        x = self.hid2img(h)
        return torch.sigmoid(x)

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparameterize = mu + sigma * epsilon
        x_reconstructed = self.decode(z_reparameterize)

        # return also mu, sigma for KL divergence loss
        return x_reconstructed, mu, sigma
