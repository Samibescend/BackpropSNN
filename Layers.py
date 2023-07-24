# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 17:24:48 2023

@author: YFGI6212
"""

import torch
import torch.nn as nn

# from torch import functional as F
import snntorch as snn

from snntorch._neurons.neurons import _SpikeTensor, _SpikeTorchConv, LIF


class RLeakyNoBias(LIF):
    
   
    def __init__(
        self,
        beta,
        V=1.0,
        all_to_all=True,
        linear_features=None,
        conv2d_channels=None,
        kernel_size=None,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        learn_recurrent=True,  # changed learn_V
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        bias = False
    ):
        super(RLeakyNoBias, self).__init__(
            beta,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
        )

        self.bias = bias
        self.all_to_all = all_to_all
        self.learn_recurrent = learn_recurrent

        # linear params
        self.linear_features = linear_features

        # Conv2d params
        self.kernel_size = kernel_size
        self.conv2d_channels = conv2d_channels

        # catch cases
        self._rleaky_init_cases()

        # initialize recurrent connections
        if self.all_to_all:  # init all-all connections
            self._init_recurrent_net()
        else:  # initialize 1-1 connections
            self._V_register_buffer(V, learn_recurrent)
            self._init_recurrent_one_to_one()

        if not learn_recurrent:
            self._disable_recurrent_grad()

        if self.init_hidden:
            self.spk, self.mem = self.init_rleaky()
        #     self.state_fn = self._build_state_function_hidden
        # else:
        #     self.state_fn = self._build_state_function


    def forward(self, input_, spk=False, mem=False):
        if hasattr(spk, "init_flag") or hasattr(
            mem, "init_flag"
        ):  # only triggered on first-pass
            spk, mem = _SpikeTorchConv(spk, mem, input_=input_)
        # init_hidden case
        elif mem is False and hasattr(self.mem, "init_flag"):
            self.spk, self.mem = _SpikeTorchConv(
                self.spk, self.mem, input_=input_
            )

        # TO-DO: alternatively, we could do torch.exp(-1 /
        # self.beta.clamp_min(0)), giving actual time constants instead of
        # values in [0, 1] as initial beta beta = self.beta.clamp(0, 1)

        if not self.init_hidden:
            self.reset = self.mem_reset(mem)
            mem = self._build_state_function(input_, spk, mem)

            if self.state_quant:
                mem = self.state_quant(mem)

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)  # batch_size
            else:
                spk = self.fire(mem)

            return spk, mem

        # intended for truncated-BPTT where instance variables are hidden
        # states
        if self.init_hidden:
            self._rleaky_forward_cases(spk, mem)
            self.reset = self.mem_reset(self.mem)
            self.mem = self._build_state_function_hidden(input_)

            if self.state_quant:
                self.mem = self.state_quant(self.mem)

            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
            else:
                self.spk = self.fire(self.mem)

            if self.output:  # read-out layer returns output+states
                return self.spk, self.mem
            else:  # hidden layer e.g., in nn.Sequential, only returns output
                return self.spk


    def _init_recurrent_net(self):
        if self.all_to_all:
            if self.linear_features:
                self._init_recurrent_linear()
            elif self.kernel_size is not None:
                self._init_recurrent_conv2d()
        else:
            self._init_recurrent_one_to_one()

    def _init_recurrent_linear(self):
        self.recurrent = nn.Linear(self.linear_features, self.linear_features,bias = self.bias)

    def _init_recurrent_conv2d(self):
        self._init_padding()
        self.recurrent = nn.Conv2d(
            in_channels=self.conv2d_channels,
            out_channels=self.conv2d_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )

    def _init_padding(self):
        if type(self.kernel_size) is int:
            self.padding = self.kernel_size // 2, self.kernel_size // 2
        else:
            self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2

    def _init_recurrent_one_to_one(self):
        self.recurrent = RecurrentOneToOneNoBias(self.V)

    def _disable_recurrent_grad(self):
        for param in self.recurrent.parameters():
            param.requires_grad = False

    def _base_state_function(self, input_, spk, mem):
        base_fn = self.beta.clamp(0, 1) * mem + input_ + self.recurrent(spk)
        return base_fn

    def _build_state_function(self, input_, spk, mem):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = self._base_state_function(
                input_, spk, mem - self.reset * self.threshold
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = self._base_state_function(
                input_, spk, mem
            ) - self.reset * self._base_state_function(input_, spk, mem)
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function(input_, spk, mem)
        return state_fn

    def _base_state_function_hidden(self, input_):
        base_fn = (
            self.beta.clamp(0, 1) * self.mem
            + input_
            + self.recurrent(self.spk)
        )
        return base_fn

    def _build_state_function_hidden(self, input_):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = (
                self._base_state_function_hidden(input_)
                - self.reset * self.threshold
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = self._base_state_function_hidden(
                input_
            ) - self.reset * self._base_state_function_hidden(input_)
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function_hidden(input_)
        return state_fn

    def _rleaky_forward_cases(self, spk, mem):
        if mem is not False or spk is not False:
            raise TypeError(
                "When `init_hidden=True`," "RLeaky expects 1 input argument."
            )

    def _rleaky_init_cases(self):
        all_to_all_bool = bool(self.all_to_all)
        linear_features_bool = self.linear_features
        conv2d_channels_bool = bool(self.conv2d_channels)
        kernel_size_bool = bool(self.kernel_size)

        if all_to_all_bool:
            if not (linear_features_bool):
                if not (conv2d_channels_bool or kernel_size_bool):
                    raise TypeError(
                        "When `all_to_all=True`, RLeaky requires either"
                        "`linear_features` or (`conv2d_channels` and "
                        "`kernel_size`) to be specified. The "
                        "shape should match the shape of the output spike of "
                        "the layer."
                    )
                elif conv2d_channels_bool ^ kernel_size_bool:
                    raise TypeError(
                        "`conv2d_channels` and `kernel_size` must both be"
                        "specified. The shape of `conv2d_channels` should "
                        "match the shape of the output"
                        "spikes."
                    )
            elif (linear_features_bool and kernel_size_bool) or (
                linear_features_bool and conv2d_channels_bool
            ):
                raise TypeError(
                    "`linear_features` cannot be specified at the same time as"
                    "`conv2d_channels` or `kernel_size`. A linear layer and "
                    "conv2d layer cannot both"
                    "be specified at the same time."
                )
        else:
            if (
                linear_features_bool
                or conv2d_channels_bool
                or kernel_size_bool
            ):
                raise TypeError(
                    "When `all_to_all`=False, none of `linear_features`,"
                    "`conv2d_channels`, or `kernel_size` should be specified. "
                    "The weight `V` is used"
                    "instead."
                )


    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended
        for use in truncated backpropagation through time where hidden state
        variables
        are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], RLeakyNoBias):
                cls.instances[layer].mem.detach_()
                cls.instances[layer].spk.detach_()



    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], RLeakyNoBias):
                (
                    cls.instances[layer].spk,
                    cls.instances[layer].mem,
                ) = cls.instances[layer].init_rleaky()




class RecurrentOneToOneNoBias(nn.Module):
    def __init__(self, V):
        super(RecurrentOneToOneNoBias, self).__init__()
        self.V = V

    def forward(self, x):
        return x * self.V  # element-wise or global multiplication