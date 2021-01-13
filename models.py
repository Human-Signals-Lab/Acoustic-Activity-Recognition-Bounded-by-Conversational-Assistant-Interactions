import os
import sys
import time
import math
import librosa
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.parameter import Parameter

class DFTBase(nn.Module):
    def __init__(self):
        """Base class for DFT and IDFT matrix"""
        super(DFTBase, self).__init__()

    def dft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W

    def idft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W


class DFT(DFTBase):
    def __init__(self, n, norm):
        """Calculate DFT, IDFT, RDFT, IRDFT. 

        Args:
          n: fft window size
          norm: None | 'ortho'
        """
        super(DFT, self).__init__()

        self.W = self.dft_matrix(n)
        self.inv_W = self.idft_matrix(n)

        self.W_real = torch.Tensor(np.real(self.W))
        self.W_imag = torch.Tensor(np.imag(self.W))
        self.inv_W_real = torch.Tensor(np.real(self.inv_W))
        self.inv_W_imag = torch.Tensor(np.imag(self.inv_W))

        self.n = n
        self.norm = norm

    def dft(self, x_real, x_imag):
        """Calculate DFT of signal. 

        Args:
          x_real: (n,), signal real part
          x_imag: (n,), signal imag part

        Returns:
          z_real: (n,), output real part
          z_imag: (n,), output imag part
        """
        z_real = torch.matmul(x_real, self.W_real) - torch.matmul(x_imag, self.W_imag)
        z_imag = torch.matmul(x_imag, self.W_real) + torch.matmul(x_real, self.W_imag)

        if self.norm is None:
            pass
        elif self.norm == 'ortho':
            z_real /= math.sqrt(self.n)
            z_imag /= math.sqrt(self.n)

        return z_real, z_imag

        return z_real, z_imag

    def idft(self, x_real, x_imag):
        """Calculate IDFT of signal. 

        Args:
          x_real: (n,), signal real part
          x_imag: (n,), signal imag part

        Returns:
          z_real: (n,), output real part
          z_imag: (n,), output imag part
        """
        z_real = torch.matmul(x_real, self.inv_W_real) - torch.matmul(x_imag, self.inv_W_imag)
        z_imag = torch.matmul(x_imag, self.inv_W_real) + torch.matmul(x_real, self.inv_W_imag)

        if self.norm is None:
            z_real /= self.n
        elif self.norm == 'ortho':
            z_real /= math.sqrt(n)
            z_imag /= math.sqrt(n)

        return z_real, z_imag

    def rdft(self, x_real):
        """Calculate right DFT of signal. 

        Args:
          x_real: (n,), signal real part
          x_imag: (n,), signal imag part

        Returns:
          z_real: (n // 2 + 1,), output real part
          z_imag: (n // 2 + 1,), output imag part
        """
        n_rfft = self.n // 2 + 1
        z_real = torch.matmul(x_real, self.W_real[..., 0 : n_rfft])
        z_imag = torch.matmul(x_real, self.W_imag[..., 0 : n_rfft])

        if self.norm is None:
            pass
        elif self.norm == 'ortho':
            z_real /= math.sqrt(self.n)
            z_imag /= math.sqrt(self.n)

        return z_real, z_imag

    def irdft(self, x_real, x_imag):
        """Calculate inverse right DFT of signal. 

        Args:
          x_real: (n // 2 + 1,), signal real part
          x_imag: (n // 2 + 1,), signal imag part

        Returns:
          z_real: (n,), output real part
          z_imag: (n,), output imag part
        """
        n_rfft = self.n // 2 + 1

        flip_x_real = torch.flip(x_real, dims=(-1,))
        x_real = torch.cat((x_real, flip_x_real[..., 1 : n_rfft - 1]), dim=-1)

        flip_x_imag = torch.flip(x_imag, dims=(-1,))
        x_imag = torch.cat((x_imag, -1. * flip_x_imag[..., 1 : n_rfft - 1]), dim=-1)

        z_real = torch.matmul(x_real, self.inv_W_real) - torch.matmul(x_imag, self.inv_W_imag)

        if self.norm is None:
            z_real /= self.n
        elif self.norm == 'ortho':
            z_real /= math.sqrt(n)

        return z_real
        

class STFT(DFTBase):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, 
        window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        """Implementation of STFT with Conv1d. The function has the same output 
        of librosa.core.stft
        """
        super(STFT, self).__init__()

        assert pad_mode in ['constant', 'reflect']

        self.n_fft = n_fft
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame
        if win_length is None:
            win_length = n_fft

        # Set the default hop, if it's not already specified
        if hop_length is None:
            hop_length = int(win_length // 4)

        fft_window = librosa.filters.get_window(window, win_length, fftbins=True)

        # Pad the window out to n_fft size
        fft_window = librosa.util.pad_center(fft_window, n_fft)

        # DFT & IDFT matrix
        self.W = self.dft_matrix(n_fft)

        out_channels = n_fft // 2 + 1

        self.conv_real = nn.Conv1d(in_channels=1, out_channels=out_channels, 
            kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, 
            groups=1, bias=False)

        self.conv_imag = nn.Conv1d(in_channels=1, out_channels=out_channels, 
            kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, 
            groups=1, bias=False)

        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, data_length)

        Returns:
          real: (batch_size, n_fft // 2 + 1, time_steps)
          imag: (batch_size, n_fft // 2 + 1, time_steps)
        """

        x = input[:, None, :]   # (batch_size, channels_num, data_length)

        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        real = self.conv_real(x)
        imag = self.conv_imag(x)
        # (batch_size, n_fft // 2 + 1, time_steps)

        real = real[:, None, :, :].transpose(2, 3)
        imag = imag[:, None, :, :].transpose(2, 3)
        # (batch_size, 1, time_steps, n_fft // 2 + 1)

        return real, imag


def magphase(real, imag):
    mag = (real ** 2 + imag ** 2) ** 0.5
    cos = real / mag
    sin = imag / mag
    return mag, cos, sin


class ISTFT(DFTBase):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, 
        window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        """Implementation of ISTFT with Conv1d. The function has the same output 
        of librosa.core.istft
        """
        super(ISTFT, self).__init__()

        assert pad_mode in ['constant', 'reflect']

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame
        if win_length is None:
            win_length = n_fft

        # Set the default hop, if it's not already specified
        if hop_length is None:
            hop_length = int(win_length // 4)

        ifft_window = librosa.filters.get_window(window, win_length, fftbins=True)

        # Pad the window out to n_fft size
        ifft_window = librosa.util.pad_center(ifft_window, n_fft)

        # DFT & IDFT matrix
        self.W = self.idft_matrix(n_fft) / n_fft

        self.conv_real = nn.Conv1d(in_channels=n_fft, out_channels=n_fft, 
            kernel_size=1, stride=1, padding=0, dilation=1, 
            groups=1, bias=False)

        self.conv_imag = nn.Conv1d(in_channels=n_fft, out_channels=n_fft, 
            kernel_size=1, stride=1, padding=0, dilation=1, 
            groups=1, bias=False)

        
        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W * ifft_window[None, :]).T)[:, :, None]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W * ifft_window[None, :]).T)[:, :, None]
        # (n_fft // 2 + 1, 1, n_fft)
        
        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, real_stft, imag_stft, length):
        """input: (batch_size, 1, time_steps, n_fft // 2 + 1)

        Returns:
          real: (batch_size, data_length)
        """

        device = next(self.parameters()).device
        batch_size = real_stft.shape[0]

        real_stft = real_stft[:, 0, :, :].transpose(1, 2)
        imag_stft = imag_stft[:, 0, :, :].transpose(1, 2)
        # (batch_size, n_fft // 2 + 1, time_steps)

        # Full stft
        full_real_stft = torch.cat((real_stft, torch.flip(real_stft[:, 1 : -1, :], dims=[1])), dim=1)
        full_imag_stft = torch.cat((imag_stft, - torch.flip(imag_stft[:, 1 : -1, :], dims=[1])), dim=1)

        # Reserve space for reconstructed waveform
        if length:
            if self.center:
                padded_length = length + int(self.n_fft)
            else:
                padded_length = length
            n_frames = min(
                real_stft.shape[2], int(np.ceil(padded_length / self.hop_length)))
        else:
            n_frames = real_stft.shape[2]
 
        expected_signal_len = self.n_fft + self.hop_length * (n_frames - 1)
        expected_signal_len = self.n_fft + self.hop_length * (n_frames - 1)
        y = torch.zeros(batch_size, expected_signal_len).to(device)

        # IDFT
        s_real = self.conv_real(full_real_stft) - self.conv_imag(full_imag_stft)

        # Overlap add
        for i in range(n_frames):
            y[:, i * self.hop_length : i * self.hop_length + self.n_fft] += s_real[:, :, i]

        ifft_window_sum = librosa.filters.window_sumsquare(self.window, n_frames,
            win_length=self.win_length, n_fft=self.n_fft, hop_length=self.hop_length)

        approx_nonzero_indices = np.where(ifft_window_sum > librosa.util.tiny(ifft_window_sum))[0]
        approx_nonzero_indices = torch.LongTensor(approx_nonzero_indices).to(device)
        ifft_window_sum = torch.Tensor(ifft_window_sum).to(device)
        
        y[:, approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices][None, :]

        # Trim or pad to length
        if length is None:
            if self.center:
                y = y[:, self.n_fft // 2 : -self.n_fft // 2]
        else:
            if self.center:
                start = self.n_fft // 2
            else:
                start = 0

            y = y[:, start : start + length]
            (batch_size, len_y) = y.shape
            if y.shape[-1] < length:
                y = torch.cat((y, torch.zeros(batch_size, length - len_y).to(device)), dim=-1)

        return y
        
class Spectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, 
        window='hann', center=True, pad_mode='reflect', power=2.0, 
        freeze_parameters=True):
        """Calculate spectrogram using pytorch. The STFT is implemented with 
        Conv1d. The function has the same output of librosa.core.stft
        """
        super(Spectrogram, self).__init__()

        self.power = power

        self.stft = STFT(n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=window, center=center, 
            pad_mode=pad_mode, freeze_parameters=True)

    def forward(self, input):
        """input: (batch_size, 1, time_steps, n_fft // 2 + 1)

        Returns:
          spectrogram: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        (real, imag) = self.stft.forward(input)
        # (batch_size, n_fft // 2 + 1, time_steps)

        spectrogram = real ** 2 + imag ** 2

        if self.power == 2.0:
            pass
        else:
            spectrogram = spectrogram ** (power / 2.0)

        return spectrogram


class LogmelFilterBank(nn.Module):
    def __init__(self, sr=32000, n_fft=2048, n_mels=64, fmin=50, fmax=14000, is_log=True, 
        ref=1.0, amin=1e-10, top_db=80.0, freeze_parameters=True):
        """Calculate logmel spectrogram using pytorch. The mel filter bank is 
        the pytorch implementation of as librosa.filters.mel 
        """
        super(LogmelFilterBank, self).__init__()

        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

        self.melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=fmax).T
        # (n_fft // 2 + 1, mel_bins)

        self.melW = nn.Parameter(torch.Tensor(self.melW))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, channels, time_steps)
        
        Output: (batch_size, time_steps, mel_bins)
        """

        # Mel spectrogram
        mel_spectrogram = torch.matmul(input, self.melW)

        # Logmel spectrogram
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram

        return output


    def power_to_db(self, input):
        """Power to db, this function is the pytorch implementation of 
        librosa.core.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise ParameterError('top_db must be non-negative')
            log_spec = torch.clamp(log_spec, min=log_spec.max() - self.top_db, max=np.inf)

        return log_spec


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size, stride=(2,2))
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock5x5, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


class FineTuneCnn6(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(FineTuneCnn6, self).__init__()
        
        self.base = Cnn65x5(sample_rate, window_size, hop_size, mel_bins, fmin,fmax, 527)
        pretrained_checkpoint_path = '/media/hd4t2/Rebecca/Research-VoiceAware-LightSense/data-analysis/PANNs-models/Cnn6_mAP=0.343.pth'
        checkpoint = torch.load(pretrained_checkpoint_path, map_location='cuda')
        self.base.load_state_dict(checkpoint['model'])
        self.base.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
        self.init_weights()

    def init_weights(self):
        init_layer(self.base.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        output_dict = self.base(input,mixup_lambda)
        embedding = output_dict['embedding']
        clipwise_output = torch.sigmoid(self.base.fc_audioset(embedding))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict

class FineTuneCnn6_FC2Classifier(nn.Module):
    ## adding a CNN classifier at the end
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(FineTuneCnn6_CNNClassifier, self).__init__()
        
        self.base = Cnn65x5(sample_rate, window_size, hop_size, mel_bins, fmin,fmax, 527)
        pretrained_checkpoint_path = '/media/hd4t2/Rebecca/Research-VoiceAware-LightSense/data-analysis/PANNs-models/Cnn6_mAP=0.343.pth'
        checkpoint = torch.load(pretrained_checkpoint_path, map_location='cuda')
        self.base.load_state_dict(checkpoint['model'])

        self.base.fc_audioset = nn.Linear(512, classes_num, bias=True)
        self.fc1 = nn.Linear(512, 256, bias = True)
        self.fc_class = nn.Linear(256, classes_num, bias = True)
        
        self.init_weights()

    def init_weights(self):
        # init_layer(self.base.fc_audioset)
        init_layer(self.fc1)
        init_layer(self.fc_class)

 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        output_dict = self.base(input,mixup_lambda)
        embedding = output_dict['embedding']
        x = F.relu_(self.fc1(x))
        embedding2 = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_class(x))

        # clipwise_output = torch.sigmoid(self.base.fc_audioset(embedding))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding2}

        return output_dict


class FineTuneCnn6_CNNClassifier(nn.Module):
    ## adding a CNN classifier at the end
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(FineTuneCnn6_CNNClassifier, self).__init__()
        
        self.base = Cnn65x5(sample_rate, window_size, hop_size, mel_bins, fmin,fmax, 527)
        pretrained_checkpoint_path = '/media/hd4t2/Rebecca/Research-VoiceAware-LightSense/data-analysis/PANNs-models/Cnn6_mAP=0.343.pth'
        checkpoint = torch.load(pretrained_checkpoint_path, map_location='cuda')
        self.base.load_state_dict(checkpoint['model'])

        self.base.fc_audioset = nn.Linear(512, classes_num, bias=True)

        self.conv1 = ConvBlock1D(in_channels = 1, out_channels = 64)
        self.conv2 = ConvBlock1D(in_channels = 64, out_channels = 128)
        self.conv3 = ConvBlock1D(in_channels = 128, out_channels = 256)

        self.fc1 = nn.Linear(256, 256, bias = True)
        self.fc_class = nn.Linear(256, classes_num, bias = True)
        
        self.init_weights()

    def init_weights(self):
        # init_layer(self.base.fc_audioset)
        init_layer(self.fc1)
        init_layer(self.fc_class)

 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        output_dict = self.base(input,mixup_lambda)
        embedding = output_dict['embedding']

        embedding = embedding[:,np.newaxis,:]
        x = self.conv1(embedding, pool_size = 2, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, pool_size=2, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, pool_size=2, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.relu_(self.fc1(x))
        embedding2 = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_class(x))

        # clipwise_output = torch.sigmoid(self.base.fc_audioset(embedding))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding2}

        return output_dict


class Cnn65x5(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn65x5, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict


class Cnn6(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        

        super(Cnn6, self).__init__()
        self.name = 'CNN6'
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict


class Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        conv_out = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(conv_out, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding, 'conv_out': conv_out}

        return output_dict

class BidirectionalGRU(nn.Module):

    def __init__(self, input_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=input_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, dropout = dropout, bidirectional=True)
        # self.bn = nn.BatchNorm1d(rnn_dim)

    def init_weights(self):
        init_layer(self.BiGRU)
        # init_bn(self.bn)

    def forward(self, x):
        self.BiGRU.flatten_parameters()
        x, _ = self.BiGRU(x)
        # x = self.bn(x)
        return x


class FineTuneCNN14_BiGRU(nn.Module):

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        super(FineTuneCNN14_BiGRU, self).__init__()

        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin,fmax, 527)
        pretrained_checkpoint_path = '../PANNs-models/Cnn14_mAP=0.431.pth'
        checkpoint = torch.load(pretrained_checkpoint_path, map_location = 'cuda')#map_location='cuda')
        self.base.load_state_dict(checkpoint['model'])

        self.biGRU = BidirectionalGRU(2, 128, .2, True)
        self.fc = nn.Linear(63, 256, bias = True)
        self.output = nn.Linear(256, classes_num, bias = True)

        self.init_weights

    def init_weights(self):
        init_layer(self.output)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        # print('input_size: {}'.format(input.size()))
        output_dict = self.base(input,mixup_lambda)
        embedding = output_dict['conv_out']

        # size = embedding.size()

        embedding1 = torch.mean(embedding, dim=1)
        (embedding2, _) = torch.max(embedding, dim = 1)

        embedding = embedding1 + embedding2

        x = self.biGRU(embedding)

        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc(x))       
        embedding = F.dropout(x, p=0.5, training=self.training)

        clipwise_output = torch.sigmoid(self.output(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict

### added just to work with loading of model in RasPi
class RaspiCNN14(nn.Module):
    def __init__(self,sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(RaspiCNN14, self).__init__()
        
        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin,fmax, classes_num)


    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        output_dict = self.base(input,mixup_lambda)
        
        return output_dict

class FineTuneCNN14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(FineTuneCNN14, self).__init__()
        
        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin,fmax, 527)
        pretrained_checkpoint_path = '../PANNs-models/Cnn14_mAP=0.431.pth'
        checkpoint = torch.load(pretrained_checkpoint_path, map_location='cuda')
        self.base.load_state_dict(checkpoint['model'])
        self.base.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weights()

    def init_weights(self):
        init_layer(self.base.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        output_dict = self.base(input,mixup_lambda)
        embedding = output_dict['embedding']
        clipwise_output = torch.sigmoid(self.base.fc_audioset(embedding))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict


class FineTuneCNN14_UpdateAll(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(FineTuneCNN14_UpdateAll, self).__init__()
        
        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin,fmax, 527)
        pretrained_checkpoint_path = '/media/hd4t2/Rebecca/Research-VoiceAware-LightSense/data-analysis/PANNs-models/Cnn14_mAP=0.431.pth'
        checkpoint = torch.load(pretrained_checkpoint_path, map_location='cuda')
        self.base.load_state_dict(checkpoint['model'])
        self.base.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weights()

    def init_weights(self):
        init_layer(self.base.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        """
        Input: (batch_size, data_length)"""

        x = self.base.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.base.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.base.bn0(x)
        x = x.transpose(1, 3)
        
        
        x = self.base.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.base.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.base.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.base.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        return output_dict


class ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvPreWavBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1,
                              padding=1, bias=False)
                              
        self.conv2 = nn.Conv1d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1, dilation=2, 
                              padding=2, bias=False)
                              
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)
        
        return x

class Wavegram_Logmel_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Wavegram_Logmel_Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        a1 = F.relu_(self.pre_bn0(self.pre_conv0(input[:, None, :])))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = torch.cat((x, a1), dim=1)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict


class FineTune_WavegramLogmelCNN14(nn.Module):

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        super(FineTune_WavegramLogmelCNN14, self).__init__()

        
        self.base = Wavegram_Logmel_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin,fmax, 527)
        pretrained_checkpoint_path = '/media/hd4t2/Rebecca/Research-VoiceAware-LightSense/data-analysis/PANNs-models/Wavegram_Logmel_Cnn14_mAP=0.439.pth'
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

        self.base.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.base.fc_audioset)


    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        output_dict = self.base(input,mixup_lambda)
        # embedding = output_dict['embedding']
        clipwise_output = torch.sigmoid(self.base.fc_audioset(output_dict['embedding']))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict



class VGGish_Cnn4(nn.Module):
    def __init__(self, classes_num):
        

        super(VGGish_Cnn4, self).__init__()
        self.name = 'VGGish_CNN4'

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)

        self.fc1 = nn.Linear(128, 128, bias=True)
        self.fc_audioset = nn.Linear(128, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        # init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        
        x = self.conv_block1(input, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1,
                              padding=1, bias=False)
                              
        self.conv2 = nn.Conv1d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1,
                              padding=1, bias=False)
                              
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=2, pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool1d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool1d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool1d(x, kernel_size=pool_size)
            x2 = F.max_pool1d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class VGGish_Cnn1D(nn.Module):
    def __init__(self, classes_num):
        

        super(VGGish_Cnn1D, self).__init__()
        self.name = 'VGGish_CNN1D'

        self.conv_block1 = ConvBlock1D(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock1D(in_channels=64, out_channels=128)

        self.fc1 = nn.Linear(128, 128, bias=True)
        self.fc_audioset = nn.Linear(128, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        # init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        
        x = self.conv_block1(input, pool_size=2, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=2, pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict

