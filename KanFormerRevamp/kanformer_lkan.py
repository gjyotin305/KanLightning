import numpy as np
import torch
import torch.nn.functional as F

from .kanformer_utils import curve2coeff, b_splines

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        grid_size=5,
        k=3,
        noise_scale=0.1,
        noise_scale_base=0.1,
        scale_spline=None,
        base_fun=torch.nn.SiLU(),
        grid_eps=0.02,
        grid_range=[-1, +1],
        bias=False,
        bias_trainable=True,
        scale_spline_trainable=True,
        scale_base_trainable=True,
        device="cpu",
    ):
        torch.nn.Module.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.base_fun = base_fun
        self.grid_eps = grid_eps
        self.size = in_dim * out_dim
        self.grid_size = grid_size
        self.device = device

        step = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-k, grid_size + k + 1, device=device) * step + grid_range[0]
        ).repeat(self.in_dim, 1)
        self.register_buffer("grid", grid)  # grid [in_dim, grid_size + 2*k + 1]

        if scale_spline is not None:
            self.scale_spline = torch.nn.Parameter(
                torch.full(
                    (
                        out_dim,
                        in_dim,
                    ),
                    fill_value=scale_spline,
                    device=device,
                ),
                requires_grad=scale_spline_trainable,
            )
        else:
            self.register_buffer("scale_spline", torch.tensor([1.0], device=device))

        noise = (
            (torch.rand(grid_size + 1, in_dim, out_dim, device=device) - 1 / 2)
            * noise_scale
            / self.grid_size  # TODO: (np.sqrt(in_dim) * np.sqrt(grid_size)) ?
        )
        self.coeff = torch.nn.Parameter(
            curve2coeff(
                x=self.grid.T[k:-k],  # [grid_size + 1, in_dim]
                y=noise,  # [grid_size + 1, in_dim, out_dim]
                grid=self.grid,
                k=k,
            ).contiguous()
        )  # [out_dim, in_dim, grid_size + k]

        self.scale_base = torch.nn.Parameter(
            (
                1 / (in_dim**0.5)
                + (torch.randn(self.out_dim, self.in_dim, device=device) * 2 - 1)
                * noise_scale_base
            ),
            requires_grad=scale_base_trainable,
        )

        if bias is True:
            self.bias = torch.nn.Parameter(
                torch.rand(out_dim), requires_grad=bias_trainable
            )
        else:
            self.bias = None

    def forward(self, x):
        shape = x.shape[:-1]
        x = x.view(-1, self.in_dim)
        # x [batch, in_dim]

        splines = b_splines(x, self.grid, self.k)  # [batch_size, in_dim, grid_size + k]

        ####### Efficient KAN forward #########

        batch_size = x.shape[0]
        y_b = F.linear(self.base_fun(x), self.scale_base)
        # [batch_size, in_dim] @ [out_dim, in_dim]^T = [batch_size, out_dim]

        y_spline = F.linear(
            splines.view(batch_size, -1),
            (self.coeff * self.scale_spline.unsqueeze(-1)).view(self.out_dim, -1),
        )  # [batch_size, in_dim * (grid_size + k)] @ [out_dim, in_dim * (grid_size + k)]^T = [batch, out_dim]

        y = y_b + y_spline

        if self.bias is not None:
            y = y + self.bias

        #######################################################################

        y = y.view(*shape, self.out_dim)

        return y

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.coeff.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

    @torch.no_grad()
    def update_grid(self, x, margin=0.01):
        batch_size = x.shape[0]

        # [batch_size, in_dim, grid_size + k]
        splines = b_splines(x, self.grid, self.k)

        # TODO: Is this correct?
        orig_coeff = self.coeff * self.scale_spline.unsqueeze(-1)

        # [in_dim, batch_size, grid_size + k] @ [in_dim, grid_size + k, out_dim] = [in_dim, batch_size, out_dim] -> [batch_size, in_dim, out_dim]
        y = (splines.permute(1, 0, 2) @ orig_coeff.permute(1, 2, 0)).permute(1, 0, 2)

        # sort activations in ascending order for each input dimension
        x_sorted = torch.sort(x, dim=0)[0]  # [batch_size, in_dim]

        # sample grid points from sorted activations
        # [grid_size + 1, in_dim]
        grid_adaptive = x_sorted[
            torch.linspace(
                0,
                batch_size - 1,
                self.grid_size + 1,
                dtype=torch.int64,
                device=self.device,
            )
        ]

        # find max and min values for each input dimension + margin and sample grid points uniformly from this range
        uniform_step = (
            x_sorted[-1] - x_sorted[0] + 2 * margin
        ) / self.grid_size  # [in_dim]
        # [grid_size + 1, in_dim]
        grid_uniform = (
            torch.arange(self.grid_size + 1, device=self.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # combine adaptive and uniform grid points
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        # grid is [grid_size + 1, in_dim] but self.grid is [in_dim, grid_size + 2*k + 1], so we need to expand grid
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.k, 0, -1, device=self.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.k + 1, device=self.device).unsqueeze(1),
            ],
            dim=0,
        )

        # set new grid, transpose [grid_size + 2*k + 1, in_dim] -> [in_dim, grid_size + 2*k + 1]
        self.grid.copy_(grid.T)
        self.coeff.data.copy_(curve2coeff(x, y, self.grid, self.k))