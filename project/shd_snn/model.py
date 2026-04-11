"""Spiking neural network model for the SHD dataset."""

import numpy as np
import torch


class SurrGradSpike(torch.autograd.Function):
    """Spiking nonlinearity with surrogate gradient.

    Uses the normalized negative part of a fast sigmoid
    as in Zenke & Ganguli (2018).
    """

    scale = 100.0

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2


spike_fn = SurrGradSpike.apply


class SHDModel:
    """Recurrent spiking network for SHD classification.

    Attributes:
        w1: Input-to-hidden weights (nb_inputs x nb_hidden)
        w2: Hidden-to-output weights (nb_hidden x nb_outputs)
        v1: Recurrent hidden weights (nb_hidden x nb_hidden)
    """

    def __init__(
        self,
        nb_inputs: int = 700,
        nb_hidden: int = 200,
        nb_outputs: int = 20,
        nb_steps: int = 100,
        time_step: float = 1e-3,
        tau_mem: float = 10e-3,
        tau_syn: float = 5e-3,
        weight_scale: float = 0.2,
        device: torch.device | None = None,
    ):
        self.nb_inputs = nb_inputs
        self.nb_hidden = nb_hidden
        self.nb_outputs = nb_outputs
        self.nb_steps = nb_steps
        self.device = device or torch.device("cpu")
        self.dtype = torch.float

        self.alpha = float(np.exp(-time_step / tau_syn))
        self.beta = float(np.exp(-time_step / tau_mem))

        self.w1 = torch.empty(
            (nb_inputs, nb_hidden), device=self.device, dtype=self.dtype, requires_grad=True
        )
        torch.nn.init.normal_(self.w1, mean=0.0, std=weight_scale / np.sqrt(nb_inputs))

        self.w2 = torch.empty(
            (nb_hidden, nb_outputs), device=self.device, dtype=self.dtype, requires_grad=True
        )
        torch.nn.init.normal_(self.w2, mean=0.0, std=weight_scale / np.sqrt(nb_hidden))

        self.v1 = torch.empty(
            (nb_hidden, nb_hidden), device=self.device, dtype=self.dtype, requires_grad=True
        )
        torch.nn.init.normal_(self.v1, mean=0.0, std=weight_scale / np.sqrt(nb_hidden))

    @property
    def params(self) -> list[torch.Tensor]:
        return [self.w1, self.w2, self.v1]

    def __call__(self, inputs: torch.Tensor, batch_size: int) -> tuple:
        return self.forward(inputs, batch_size)

    def forward(self, inputs: torch.Tensor, batch_size: int) -> tuple:
        """Run the SNN forward pass.

        Args:
            inputs: Dense input tensor (batch_size x nb_steps x nb_inputs)
            batch_size: Batch size (needed to initialize state tensors)

        Returns:
            (output_recordings, (mem_recordings, spike_recordings))
        """
        syn = torch.zeros((batch_size, self.nb_hidden), device=self.device, dtype=self.dtype)
        mem = torch.zeros((batch_size, self.nb_hidden), device=self.device, dtype=self.dtype)

        mem_rec = []
        spk_rec = []

        out = torch.zeros((batch_size, self.nb_hidden), device=self.device, dtype=self.dtype)
        h1_from_input = torch.einsum("abc,cd->abd", (inputs, self.w1))
        for t in range(self.nb_steps):
            h1 = h1_from_input[:, t] + torch.einsum("ab,bc->ac", (out, self.v1))
            mthr = mem - 1.0
            out = spike_fn(mthr)
            rst = out.detach()

            new_syn = self.alpha * syn + h1
            new_mem = (self.beta * mem + syn) * (1.0 - rst)

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)

        # Readout layer
        h2 = torch.einsum("abc,cd->abd", (spk_rec, self.w2))
        flt = torch.zeros((batch_size, self.nb_outputs), device=self.device, dtype=self.dtype)
        out = torch.zeros((batch_size, self.nb_outputs), device=self.device, dtype=self.dtype)
        out_rec = [out]
        for t in range(self.nb_steps):
            new_flt = self.alpha * flt + h2[:, t]
            new_out = self.beta * out + flt

            flt = new_flt
            out = new_out

            out_rec.append(out)

        out_rec = torch.stack(out_rec, dim=1)
        return out_rec, (mem_rec, spk_rec)

    def save(self, path: str):
        torch.save({"w1": self.w1, "w2": self.w2, "v1": self.v1}, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.w1 = checkpoint["w1"].to(self.device).requires_grad_(True)
        self.w2 = checkpoint["w2"].to(self.device).requires_grad_(True)
        self.v1 = checkpoint["v1"].to(self.device).requires_grad_(True)
