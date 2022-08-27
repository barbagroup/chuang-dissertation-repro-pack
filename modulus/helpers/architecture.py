#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""RBF net.
"""
# pylint: disable=invalid-name, no-member, too-many-arguments
from typing import Dict
from typing import List
from typing import Optional

import torch
from torch import nn
from torch import Tensor
from modulus.architecture import layers
from modulus.arch import Arch
from modulus.key import Key


class RadialBasisArch(Arch):
    """Radial Basis Neural Network that fixed scaling issues in NVIDIA's implementation.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list
    output_keys : List[Key]
        Output key list
    bounds : Dict[str, Tuple[float, float]]
        Bounds to to randomly generate radial basis functions in.
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    nr_centers : int = 128
        number of radial basis functions to use.
    sigma : float = 0.1
        Sigma in radial basis kernel.
    """

    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        bounds: Dict[str, List[float]],
        detach_keys: Optional[List[Key]] = None,
        sigma: float = 0.1,
        train_rbfs: bool = False,
        centers: Optional[List] = None,
    ) -> None:

        # make sure inpuy keys and bounds match
        assert set(key.name for key in input_keys) == set(bounds.keys())

        if detach_keys is None:
            detach_keys = []

        super().__init__(input_keys=input_keys, output_keys=output_keys, detach_keys=detach_keys)

        out_features = sum(self.output_key_dict.values())

        # use list because dict was not gaurantee to be ordered in earlier Python
        self.bounds = [bounds[key.name] for key in input_keys]

        # centers are not trainable
        centers = torch.tensor(centers, requires_grad=False, dtype=torch.float32)
        self.nr_centers = centers.shape[0]
        self.register_buffer("centers", centers)

        # scale canters
        with torch.no_grad():
            for idx, key in enumerate(input_keys):
                if key.scale is not None:
                    self.centers[:, idx] = (self.centers[:, idx] - key.scale[0]) / key.scale[1]

        # radius are trainable
        if train_rbfs:
            # calculate average radius
            sigma = 1.
            for bound in self.bounds:  # calculating the total volume
                sigma *= (bound[1] - bound[0])
            sigma /= self.centers.shape[0]  # average volume
            sigma = sigma**(1./len(bounds))  # length
            self.sigma = nn.Parameter(torch.full((self.centers.shape[0],), sigma), requires_grad=True)
        else:
            self.sigma = sigma

        self.fc_layer = layers.FCLayer(self.nr_centers, out_features, activation_fn=layers.Activation.IDENTITY)

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Override forward.
        """
        # returned shape: (npts, ndim)
        x = self.prepare_input(
            in_vars,
            self.input_key_dict.keys(),
            self.detach_key_dict,
            -1,
            self.input_scales,
            self.periodicity
        )

        # shape: (npts, nc, ndim)
        shape = (x.size(0), self.nr_centers, x.size(1))

        # returned shape: (npts, nc, ndim)
        x = x.unsqueeze(1).expand(shape)

        # return shape: (npts, nc, ndim)
        centers = self.centers.unsqueeze(0).expand(shape)

        # return shape: (npts, nc) due to the norm is only applied to the last dimension
        dist2 = ((centers - x)**2).sum(dim=-1)

        # return shape: (npts, nc)
        diam = (self.sigma.unsqueeze(0).expand((shape[0], shape[1]))**2) * 2.

        # return shape: (npts, nc)
        outs = torch.exp(-dist2/diam)

        # return shape: (npts, nout)
        x = self.fc_layer(outs)

        return self.prepare_output(x, self.output_key_dict, -1)
