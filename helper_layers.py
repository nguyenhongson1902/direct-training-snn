import torch
import torch.nn as nn
import torch.nn.functional as F


# Hyperparameters
steps = 2 # The number of timesteps
aa = 0.5 # The peak width of the approximation derivative
Vth = 0.4 # Threshold
tau = 0.25 # Decay factor


class SpikeAct(torch.autograd.Function):
    """ 
        Implementation of the spiking activation function (a step function) with an approximation of 
        gradient (a rectangular function)
    """
    @staticmethod
    def forward(ctx, input):
        """
            Implements of a step function
            Args:
                input (Tensor): Input of the function.
            Returns:
                output (Tensor): contains event-spikes, which are 0s if no events and 1s if there are spikes.
        """
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = input.clone()
        output = torch.gt(output, Vth)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        """
            Computes derivative of the step function, which is a Dirac function.
            Args:
                grad_output (Tensor): The derivative of the loss function w.r.t the output.
            Returns:
                (Tensor): The derivative of the loss function w.r.t the input.
        """
        input, = ctx.saved_tensors 
        grad_input = grad_output.clone()
        # hu is an approximated function (a rectangular function) of dg/du
        # (g function in the paper https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full#B41)
        hu = abs(input) < aa #
        hu = hu.float() / (2 * aa)
        return grad_input * hu

# A quick way to initialize the activation funtion
spikeAct = SpikeAct.apply



def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n):
    '''
        State update algorithm in the paper https://arxiv.org/abs/1809.05793.
        Affine (linear) transformation W_mul_o_t1_n (pre-synaptic inputs) is done in the tdBatchNorm layer.
    Args:
        u_t_n1 (Tensor): previous membrane potential.
        o_t_n1 (Tensor): previous spike output.
        W_mul_o_t1_n (Tensor): pre-synaptic inputs (o_t1_n is current spike input and W is the weight matrix).
    Returns:
        u_t1_n1 (Tensor): next membrane potential
        o_t1_n1 (Tensor): next spike output
    '''
    u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
    o_t1_n1 = spikeAct(u_t1_n1)
    return u_t1_n1, o_t1_n1


class tdLayer(nn.Module):
    """
        Converts a common layer to the time domain. The input tensor needs to have an additional time dimension, 
        which in this case is on the last dimension of the data. During the forward pass, a normal layer is 
        performed for each time step of the data in that time dimension.
    Args:
        layer (nn.Module): The layer needs to be converted.
        bn (nn.Module): If Batch Normalization (BN) is needed, the BN layer should be passed in together as a parameter.
    """
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = layer
        self.bn = bn

    def forward(self, x):
        """
            Applies forward function to the input tensor x through the number of timesteps and uses BN (if any).
            Args:
                x (Tensor): Input of the layer.
            Returns:
                x_ (Tensor): Output after adding the time dimension and BN (if any).
        """
        # shape of x: [N, C, H, W, T]. For example, x[..., 0] has timestep=0 and shape [N, C, H, W]
        x_ = torch.zeros(self.layer(x[..., 0]).shape + (steps,), device=x.device) # Add time dimension to the last dim
        # x_: [N, C, H, W, steps], all zeros
        for step in range(steps):
            x_[..., step] = self.layer(x[..., step])

        if self.bn is not None:
            x_ = self.bn(x_)
        return x_

        
class LIFSpike(nn.Module):
    """
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. 
        The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """
    def __init__(self):
        super(LIFSpike, self).__init__()

    def forward(self, x):
        """
            Args:
                x (Tensor), shape: [N, C, H, W, T]: Represents input of the activation function.
            Returns:
                out (Tensor), shape: [N, C, H, W, T]: Output tensor after #timesteps.
        """
        # x.shape[:-1], except the last dimension
        u   = torch.zeros(x.shape[:-1] , device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(steps):
            u, out[..., step] = state_update(u, out[..., max(step-1, 0)], x[..., step])
        return out


class tdBatchNorm(nn.BatchNorm2d):
    """
        Implementation of tdBN. Link to related paper: https://arxiv.org/abs/2011.05280. In a nutshell, it is averaged over the time 
        domain as well when doing BN. Additional scaling factors Vth and alpha are also introduced. 
        BatchNorm2d PyTorch implementation: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html.
        More information about standard Batch Normalization: http://d2l.ai/chapter_convolutional-modern/batch-norm.html#equation-eq-batchnorm.
    Args:
        num_features (int): C from an expected input of size (N, C, H, W)
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5
        momentum (float): The value used for the running_mean and running_var computation. 
                        Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
        alpha (float): An addtional parameter which may be changed in the customized Basic Block.
        affine (bool): A boolean value that when set to True, this module has learnable affine parameters. 
                        Default: True
        track_running_stats (bool): A boolean value that when set to True, this module tracks the running mean and variance, 
                                    and when set to False, this module does not track such statistics, and initializes statistics 
                                    buffers running_mean and running_var as None. When these buffers are None, this module always 
                                    uses batch statistics. in both training and eval modes. Default: True
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

        # Modify the shape of self.running_mean, self.running_var to be suitable with the mean and var shapes
        self.running_mean = torch.reshape(self.running_mean, (1, -1, 1, 1, 1))
        self.running_var = torch.reshape(self.running_var, (1, -1, 1, 1, 1))

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates (i.e. running mean and running var)
        if self.training: # In training mode, the current mean and variance are used
            mean = input.mean(dim=(0, 2, 3, 4), keepdim=True)
            # use biased var in train
            var = input.var(dim=(0, 2, 3, 4), unbiased=False, keepdim=True) # compute variance via the biased estimator (i.e. unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = self.running_mean.view_as(mean)
                self.running_var = self.running_var.view_as(var)

                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else: # In test mode, use mean and variance obtained by moving average
            mean = self.running_mean
            var = self.running_var


        input = self.alpha * Vth * (input - mean) / (torch.sqrt(var + self.eps)) # input now is x_k in the paper https://arxiv.org/pdf/2011.05280
        if self.affine: # if True, we use the affine transformation (linear transformation)
            input = input * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None] # input now is y_k in the paper https://arxiv.org/pdf/2011.05280
            # input = input * self.weight + self.bias

        return input