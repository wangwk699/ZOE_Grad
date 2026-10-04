import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np
import pdb
import math
CLIPMIN = 1e-05

def truncated_normal_z2_expectation(C: float=3.0) -> float:
    phi_c = 0.5 * (1.0 + math.erf(C / math.sqrt(2.0)))
    denom = (2.0 * phi_c - 1.0) * math.sqrt(2.0 * math.pi)
    return 1.0 - 6.0 * math.exp(-(C * C) / 2.0) / denom
TRUNCATED_NORMAL_C = truncated_normal_z2_expectation(3.0)

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

class roundSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for rounding operation.
    Forward: round(x)
    Backward: gradient = 1 (identity)
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class UniformAffineQuantizer(nn.Module):

    def __init__(self, n_bits: int=8, symmetric: bool=False, per_channel_axes=[], metric='minmax', dynamic=False, dynamic_method='per_cluster', group_size=None, shape=None, lwc=False, disable_zero_point=True, delta=None, t=None, method=None, use_sum=None):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        assert 2 <= n_bits <= 16, 'bitwidth not supported'
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -2 ** (n_bits - 1)
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** n_bits - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None
        self.scale = None
        self.zero_point = None
        self.round_zero_point = None
        self.descale = None
        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        init_value = 4.0
        if lwc:
            if group_size:
                dim1 = int(shape[0] * math.ceil(shape[1] / group_size))
                self.deficiency = shape[-1] % group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
        self.sigmoid = nn.Sigmoid()
        self.enable = True
        self.group_size = group_size
        self.delta = delta
        self.method = method
        self.t = t
        self.use_sum = use_sum
        if method not in (None, 'STE', 'HTGE', 'Uniform', 'Normal'):
            raise ValueError(f'Unsupported gradient estimator: {method}')
        if method == 'Uniform':
            self.round_module = UniformModule(delta, use_sum)
        elif method == 'Normal':
            self.round_module = NormalModule(delta, use_sum)
        elif method == 'HTGE':
            self.round_module = HTGEModule(t)
        else:
            self.round_module = RoundSTE()

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -2 ** (n_bits - 1)
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** n_bits - 1

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0], self.deficiency), dtype=x.dtype, device=x.device)
            x = torch.cat((x, pad_zeros), dim=1)
        if self.descale is not None:
            eff_scale = scale + self.descale
        else:
            eff_scale = scale
        eff_scale = eff_scale.clamp(min=0.01, max=10000.0)
        if self.group_size:
            assert len(x.shape) == 2, 'only support linear layer now'
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        normalized_x = x * (1.0 / eff_scale)
        x_int = self.round_module.forward(normalized_x)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        if self.descale is not None:
            x_dequant = x_dequant.mul(eff_scale)
        else:
            x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:, :-self.deficiency]
        return x_dequant

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == 'fix0to1':
            return x.mul_(2 ** self.n_bits - 1).round_().div_(2 ** self.n_bits - 1)
        if self.dynamic_method == 'per_token' or self.dynamic_method == 'per_channel':
            self.per_token_dynamic_calibration(x.detach())
        else:
            raise NotImplementedError()
        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        return x_dequant

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1, self.group_size)
            else:
                pad_zeros = torch.zeros((x.shape[0], self.deficiency), dtype=x.dtype, device=x.device)
                x = torch.cat((x, pad_zeros), dim=1)
                x = x.reshape(-1, self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax = x.amax(reduce_shape, keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor) * xmax
            xmin = self.sigmoid(self.lowbound_factor) * xmin
        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / (2 ** (self.n_bits - 1) - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=10000.0)
            zero_point = (2 ** (self.n_bits - 1) - 1) * torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            scale = range / (2 ** self.n_bits - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=10000.0)
            zero_point = -xmin / self.scale
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-10000.0, max=10000.0).round()

    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        descale = torch.zeros_like(self.scale)
        self.descale = nn.Parameter(descale)
        del self.scale
        del self.round_zero_point

class Uniform(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, delta, use_sum):
        out = torch.round(x)
        ctx.save_for_backward(x)
        ctx.delta = delta
        ctx.use_sum = use_sum
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：使用显式表达式计算代理梯度
        """
        x, = ctx.saved_tensors
        delta = ctx.delta
        use_sum = ctx.use_sum
        C = math.sqrt(3)
        if delta <= 1.0 / (2.0 * C):
            _lambda = 3.0
            b = torch.round(x - 0.5) + 0.5
            v = x - b
            abs_v = torch.abs(v)
            M = _lambda
            condition = abs_v < delta * math.sqrt(M)
            grad_input = torch.zeros_like(x)
            if condition.any():
                v_squared_over_delta_squared = abs_v * abs_v / (delta * delta)
                explicit_grad = 1 / (4 * math.sqrt(_lambda) * delta) * (M - v_squared_over_delta_squared)
                grad_input = torch.where(condition, explicit_grad, grad_input)
        else:
            original_shape = x.shape
            x_flat = x.view(-1)
            lambda_val = math.sqrt(3)
            search_radius = lambda_val * delta
            max_k_count = 2 * int(math.ceil(search_radius)) + 1
            k0 = torch.round(x_flat - 0.5)
            half_range = max_k_count // 2
            offsets = torch.arange(-half_range, half_range + 1, device=x.device, dtype=torch.float32)
            k_candidates = k0.unsqueeze(1) + offsets
            b_candidates = k_candidates + 0.5
            x_expanded = x_flat.unsqueeze(1)
            distances = torch.abs(x_expanded - b_candidates)
            mask = distances < search_radius
            grad_contrib = torch.zeros_like(distances)
            with torch.no_grad():
                squared_distances = (x_expanded - b_candidates) ** 2
                candidate_contrib = 3 - squared_distances / delta ** 2
                grad_contrib = torch.where(mask, candidate_contrib, torch.zeros_like(candidate_contrib))
            sum_contrib = grad_contrib.sum(dim=1)
            coeff = 1.0 / (4.0 * lambda_val * delta)
            grad_est = coeff * sum_contrib
            grad_input = grad_est.view(original_shape)
        return (grad_input * grad_output, None, None)

class RoundSTE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return round_ste(x)

    def extra_repr(self):
        return 'round_ste'

class UniformModule(nn.Module):

    def __init__(self, delta, use_sum):
        super().__init__()
        self.delta = delta
        self.use_sum = use_sum

    def forward(self, x):
        return Uniform.apply(x, self.delta, self.use_sum)

    def extra_repr(self):
        return f'Uniform(delta={self.delta} use_sum={self.use_sum})'

class NormalModule(nn.Module):

    def __init__(self, delta, use_sum):
        super().__init__()
        self.delta = delta
        self.use_sum = use_sum

    def forward(self, x):
        return Normal.apply(x, self.delta, self.use_sum)

    def extra_repr(self):
        return f'Normal(delta={self.delta} use_sum={self.use_sum})'

class HTGEModule(nn.Module):

    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x):
        return HTGE.apply(x, self.t)

    def extra_repr(self):
        return f'HTGE(delta={self.t})'

class Normal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, delta, use_sum):
        out = torch.round(x)
        ctx.save_for_backward(x)
        ctx.delta = delta
        ctx.use_sum = use_sum
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：使用正态分布近似计算代理梯度
        
        公式对应：
        给定公式：I = 1 / (2Φ(1/(2δ)) - 1) * 1/(δ√(2π)) * [exp(-(u - s(u))²/(2δ²)) - exp(-1/(8δ²))]
        
        其中：
        - Φ是标准正态分布的累积分布函数(CDF)
        - δ是正态分布的标准差
        - s(u) = round(u - 0.5) + 0.5，即最近的半整数点
        - u是输入值x
        
        推导过程：
        1. 令C = 1/(2δ)
        2. 计算Φ(C)，即标准正态分布在C处的累积概率
        3. 计算归一化因子：1/(2Φ(C) - 1)
        4. 计算高斯核部分：1/(δ√(2π)) * exp(-(x - s(x))²/(2δ²))
        5. 减去截断项：- 1/(δ√(2π)) * exp(-C²/2)
           = - 1/(δ√(2π)) * exp(-1/(8δ²))
        
        最终梯度 = 归一化因子 * (高斯核 - 截断项)

        Proposition 4.2: 截断正态的 E[z^2]=c != 1，需 g_delta = g / c（见 TRUNCATED_NORMAL_C）。
        
        Args:
            grad_output: 上游梯度
        Returns:
            grad_input: 输入x的梯度
            None: delta的梯度（不计算）
        """
        x, = ctx.saved_tensors
        delta = ctx.delta
        use_sum = ctx.use_sum
        norm_c = TRUNCATED_NORMAL_C
        C = 3
        if delta <= 1.0 / (2.0 * C):
            s_u = torch.round(x - 0.5) + 0.5
            C = torch.tensor(3.0, device=x.device, dtype=x.dtype)
            Phi_C = 0.5 * (1.0 + torch.erf(C / torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))))
            normalization_factor = 1.0 / (2.0 * Phi_C - 1.0)
            gaussian_kernel = torch.exp(-(x - s_u) ** 2 / (2.0 * delta ** 2))
            truncation_term = torch.exp(-C ** 2 / 2.0)
            normalizing_constant = 1.0 / (delta * math.sqrt(2.0 * math.pi))
            grad_input = normalization_factor * normalizing_constant * (gaussian_kernel - truncation_term)
            grad_input = grad_input / norm_c
            grad_input = grad_input * grad_output
        else:
            original_shape = x.shape
            x_flat = x.view(-1)
            C = 3.0
            search_radius = C * delta
            max_k_count = 2 * int(math.ceil(2 * search_radius)) + 1
            max_k_count = max(max_k_count, 1)
            k0 = torch.round(x_flat - 0.5)
            half_range = max_k_count // 2
            offsets = torch.arange(-half_range, half_range + 1, device=x.device, dtype=torch.float32)
            k_candidates = k0.unsqueeze(1) + offsets
            b_candidates = k_candidates + 0.5
            x_expanded = x_flat.unsqueeze(1)
            distances = torch.abs(x_expanded - b_candidates)
            mask = distances < search_radius
            grad_contrib = torch.zeros_like(distances)
            Phi_C_value = 0.5 * (1.0 + torch.erf(torch.tensor(C / math.sqrt(2.0), device=x.device, dtype=x.dtype)))
            normalization_factor = 1.0 / (2.0 * Phi_C_value - 1.0)
            normalizing_constant = 1.0 / torch.tensor(delta * math.sqrt(2.0 * math.pi), device=x.device, dtype=x.dtype)
            truncation_term = torch.exp(-torch.tensor(C ** 2 / 2.0, device=x.device, dtype=x.dtype))
            with torch.no_grad():
                squared_distances = (x_expanded - b_candidates) ** 2
                gaussian_kernels = torch.exp(-squared_distances / (2.0 * delta ** 2))
                candidate_contrib = gaussian_kernels - truncation_term
                grad_contrib = torch.where(mask, candidate_contrib, torch.zeros_like(candidate_contrib))
            sum_contrib = grad_contrib.sum(dim=1)
            coeff = normalization_factor * normalizing_constant
            grad_est = coeff * sum_contrib / norm_c
            grad_input = grad_est.view(original_shape)
            grad_input = grad_input * grad_output
        return (grad_input, None, None)

class HTGE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, t):
        out = torch.round(x)
        ctx.save_for_backward(x)
        ctx.t = t
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：使用HTGE近似梯度公式计算代理梯度
        
        公式对应：
        根据公式(24): H(x) = (a+b)/2 + (1/2)*tanh(t*(x - (a+b)/2))
        根据公式(25): ∂round(x)/∂x ≈ ∂H(x)/∂x = 1/2 * (1 - tanh^2(t*(x - (a+b)/2)))
        
        推导过程：
        1. 令 u = t*(x - (a+b)/2)
        2. 则 H(x) = (a+b)/2 + (1/2)*tanh(u)
        3. ∂H/∂x = (1/2) * ∂tanh(u)/∂x
        4. ∂tanh(u)/∂x = (1 - tanh^2(u)) * ∂u/∂x
        5. ∂u/∂x = t
        6. 所以 ∂H/∂x = (1/2) * (1 - tanh^2(u)) * t
        7. 因此梯度公式为: t/2 * (1 - tanh^2(t*(x - (a+b)/2)))
        
        Args:
            grad_output: 上游梯度
        Returns:
            grad_input: 输入x的梯度
            None: t的梯度（不计算）
        """
        x, = ctx.saved_tensors
        t = ctx.t
        a = torch.floor(x)
        b = torch.ceil(x)
        mid = (a + b) / 2.0
        u = t * (x - mid)
        tanh_u = torch.tanh(u)
        tanh_u_squared = tanh_u * tanh_u
        grad_input = t / 2.0 * (1.0 - tanh_u_squared)
        grad_input = grad_input * grad_output
        return (grad_input, None)
