import torch


class TimeDistorter:

    def __init__(
        self,
        train_distortion,
        sample_distortion,
        mu=0,
        sigma=1,
        s=-0.54,
    ):
        self.train_distortion = train_distortion  # used for sample_ft  identity
        self.sample_distortion = sample_distortion  # used for get_ft  identity
        self.mu = mu
        self.sigma = sigma
        self.s = s
        print(
            f"TimeDistorter: train_distortion={train_distortion}, sample_distortion={sample_distortion}, mu={mu}, sigma={sigma}, s={s}"
        )
        self.f_inv = None

    def train_ft(self, batch_size, device):
        if self.train_distortion == "logitnormal":
            mean = self.mu
            std = self.sigma
            normal_sample = mean + std * torch.randn((batch_size, 1), device=device)
            t_distort = torch.sigmoid(normal_sample)
        else:  # identity
            t_uniform = torch.rand((batch_size, 1), device=device)
            t_distort = self.apply_distortion(t_uniform, self.train_distortion)

        return t_distort

    def sample_ft(self, t, sample_distortion):  # identity
        t_distort = self.apply_distortion(t, sample_distortion)
        return t_distort

    def apply_distortion(self, t, distortion_type):
        assert torch.all((t >= 0) & (t <= 1)), "t must be in the range (0, 1)"

        if distortion_type == "identity":
            ft = t
        elif distortion_type == "mode":  # 带有一个参数 s 的变形，强调中段或边缘
            ft = 1 - t - self.s * (torch.cos(torch.pi / 2 * t) ** 2 - 1 + t)
            ft = torch.clamp(ft, 0.0, 1.0)
        elif distortion_type == "cos":
            ft = (1 - torch.cos(t * torch.pi)) / 2
        elif distortion_type == "revcos":
            ft = 2 * t - (1 - torch.cos(t * torch.pi)) / 2
        elif distortion_type == "polyinc":
            ft = t**2
        elif distortion_type == "polydec":
            ft = 2 * t - t**2
        elif distortion_type == "polydec_1p4":
            ft = 2 * t - t ** (1.4)
        elif distortion_type == "polydec_1p6":
            ft = 2 * t - t ** (1.6)
        elif distortion_type == "polydec_1p8":
            ft = 2 * t - t ** (1.8)
        elif distortion_type == "polydec_1p9":
            ft = 2 * t - t ** (1.9)
        elif distortion_type == "polydec_2p1":
            ft = 2 * t - t ** (2.1)
        elif distortion_type == "polydec_2p2":
            ft = 2 * t - t ** (2.2)
        elif distortion_type == "polydec_2p3":
            ft = 2 * t - t ** (2.3)
        elif distortion_type == "polydec_2p4":
            ft = 2 * t - t ** (2.4)
        elif distortion_type == "polydec_2p6":
            ft = 2 * t - t ** (2.6)
        elif distortion_type == "polydec_2p8":
            ft = 2 * t - t ** (2.8)
        # elif distortion_type == "adaptive":
        else:
            raise ValueError(f"Unknown distortion type: {distortion_type}")

        return ft
