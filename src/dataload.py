import os
import re
import glob
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchaudio
import random


# 返回Mel时频谱图
class AudioDataset(Dataset):
    def __init__(self,
                 root_dir,
                 n_mels=64,
                 n_frames=5,
                 n_fft=1024,
                 hop_length=512,
                 power=2.0,
                 stage='train',
                 num_classes=10
                 ):
        """
        初始化音频数据集

        参数:
        root_dir (str): 数据根目录，包含train和test子目录
        n_mels (int): Mel滤波器组数量
        n_frames (int): 堆叠帧数
        n_fft (int): FFT窗口大小
        hop_length (int): 帧移
        power (float): 功率谱指数
        num_classes (int): 设备ID类别数，用于独热编码
        """
        self.n_mels = n_mels # mel滤波器组数量
        self.n_frames = n_frames # 堆叠帧数（将多少帧特征堆成一个特征向量）
        self.n_fft = n_fft # FFT窗口大小
        self.hop_length = hop_length # 滑窗大小
        self.power = power # 功率谱指数
        self.num_classes = num_classes # 设备ID类别数
        self.dims = n_mels * n_frames # 特征向量的维度=mel滤波器组数量*堆叠帧数

        # 收集所有音频文件路径
        self.audio_files = []

        split_dir = os.path.join(root_dir, stage) # data/dataset/fan/train
        assert os.path.exists(split_dir), f"Path {split_dir} does not exist"
        if os.path.exists(split_dir):
            wav_files = glob.glob(os.path.join(split_dir, "*.wav")) # data/dataset/fan/train/normal/*.wav
            self.audio_files.extend(wav_files)

    def __len__(self):
        return len(self.audio_files) # 数据集样本数量即为文件个数

    def __getitem__(self, idx):
        # 获取音频文件路径
        file_path = self.audio_files[idx]

        # 提取设备状态和设备ID
        filename = os.path.basename(file_path)
        # 解析normal/anomaly
        status = 1 if 'anomaly' in filename else 0
        # 解析设备ID
        id_match = re.search(r'id_(\d+)', filename) # \d 数字 \d+后面的一串数字
        device_id = int(id_match.group(1)[-1]) if id_match else -1 # id_match.group(1)返回第 1 个括号分组（(\d+)）匹配到的内容

        # 转换设备ID为独热编码
        device_one_hot = torch.zeros(self.num_classes)
        if 0 <= device_id < self.num_classes:
            device_one_hot[device_id] = 1

        # 转换音频为特征向量
        vectors = self.file_to_vectors(file_path)

        # 转换为torch tensor
        features = torch.FloatTensor(vectors)
        features = features.unsqueeze(0)
        label = torch.LongTensor([status])
        device = device_one_hot

        return features, label, device

    def file_to_vectors(self, file_name):
        """
        参考 common.py 中的 file_to_vectors 函数
        将音频文件转换为特征向量

        参数:
        file_name (str): 音频文件路径

        返回:
        numpy.array: 特征向量数组
        """
        # 生成梅尔频谱图
        y, sr = librosa.load(file_name, sr=None, mono=True)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=self.power
        )

        # 转换为对数梅尔能量
        log_mel_spectrogram = 20.0 / self.power * np.log10(
            np.maximum(mel_spectrogram, np.finfo(np.float32).eps)
        )

        # 计算向量总数
        n_vectors = len(log_mel_spectrogram[0, :]) - self.n_frames + 1

        # 跳过太短的音频片段
        if n_vectors < 1:
            return np.empty((0, self.dims))

        # 通过连接多帧生成特征向量
        vectors = np.zeros((n_vectors, self.dims))

        for t in range(self.n_frames):
            vectors[:, self.n_mels * t: self.n_mels * (t + 1)] = \
                log_mel_spectrogram[:, t: t + n_vectors].T

        return vectors


# 返回波形
class AudioDataset_2(Dataset):
    def __init__(self,
                 root_dir,
                 stage='train',
                 num_classes=10
                 ):
        """
        初始化音频数据集

        参数:
        root_dir (str): 数据根目录，包含train和test子目录
        n_mels (int): Mel滤波器组数量
        n_frames (int): 堆叠帧数
        n_fft (int): FFT窗口大小
        hop_length (int): 帧移
        power (float): 功率谱指数
        num_classes (int): 设备ID类别数，用于独热编码
        """

        self.num_classes = num_classes # 设备ID类别数


        # 收集所有音频文件路径
        self.audio_files = []

        split_dir = os.path.join(root_dir, stage) # data/dataset/fan/train
        assert os.path.exists(split_dir), f"Path {split_dir} does not exist"
        if os.path.exists(split_dir):
            wav_files = glob.glob(os.path.join(split_dir, "*.wav")) # data/dataset/fan/train/normal/*.wav
            self.audio_files.extend(wav_files)

    def __len__(self):
        return len(self.audio_files) # 数据集样本数量即为文件个数

    def __getitem__(self, idx):
        # 获取音频文件路径
        file_path = self.audio_files[idx]

        # 提取设备状态和设备ID
        filename = os.path.basename(file_path)
        # 解析normal/anomaly
        status = 1 if 'anomaly' in filename else 0
        # 解析设备ID
        id_match = re.search(r'id_(\d+)', filename) # \d 数字 \d+后面的一串数字
        device_id = int(id_match.group(1)[-1]) if id_match else -1 # id_match.group(1)返回第 1 个括号分组（(\d+)）匹配到的内容

        # 转换设备ID为独热编码
        device_one_hot = torch.zeros(self.num_classes)
        if 0 <= device_id < self.num_classes:
            device_one_hot[device_id] = 1

        # 转换音频为特征向量
        vectors = self.file_to_vectors(file_path)

        # 转换为torch tensor
        features = torch.FloatTensor(vectors)
        features = features.unsqueeze(0)
        label = torch.LongTensor([status])
        device = device_one_hot

        return features, label, device, file_path

    def file_to_vectors(self, file_name):
        """
        参考 common.py 中的 file_to_vectors 函数
        将音频文件转换为特征向量

        参数:
        file_name (str): 音频文件路径

        返回:
        numpy.array: 特征向量数组
        """
        # 返回波形数据
        y, sr = librosa.load(file_name, sr=None, mono=True)
        return y


# 返回波形
class AudioDataset_3(Dataset):
    def __init__(self,

                 num_classes=10,
                 n_mels=64,
                 n_frames=5,
                 n_fft=1024,
                 hop_length=512,
                 power=2.0,

                 file_name_list=None,
                 pseudo_labels=None, # 接收伪标签和file_name_list 参数
                 ):
        """
        初始化音频数据集

        参数:
        root_dir (str): 数据根目录，包含train和test子目录
        n_mels (int): Mel滤波器组数量
        n_frames (int): 堆叠帧数
        n_fft (int): FFT窗口大小
        hop_length (int): 帧移
        power (float): 功率谱指数
        num_classes (int): 设备ID类别数，用于独热编码
        """

        self.num_classes = num_classes # 设备ID类别数

        self.n_mels = n_mels  # mel滤波器组数量
        self.n_frames = n_frames  # 堆叠帧数（将多少帧特征堆成一个特征向量）
        self.n_fft = n_fft  # FFT窗口大小
        self.hop_length = hop_length  # 滑窗大小
        self.power = power  # 功率谱指数

        self.dims = n_mels * n_frames  # 特征向量的维度=mel滤波器组数量*堆叠帧数


        # 收集所有音频文件路径
        self.audio_files = file_name_list
        # 获取伪标签路径
        self.pseudo_labels = pseudo_labels

        assert len(self.audio_files) > 0, "file_name_list is empty"
        assert len(self.pseudo_labels) > 0, "file_name_list is empty"

    def __len__(self):
        return len(self.audio_files) # 数据集样本数量即为文件个数

    def __getitem__(self, idx):
        # 获取音频文件路径
        file_path = self.audio_files[idx]
        # 获取伪标签
        pseudo_label = self.pseudo_labels[idx]


        # 提取设备状态和设备ID
        filename = os.path.basename(file_path)
        # 解析normal/anomaly
        status = 1 if 'anomaly' in filename else 0
        # 解析设备ID
        id_match = re.search(r'id_(\d+)', filename) # \d 数字 \d+后面的一串数字
        device_id = int(id_match.group(1)[-1]) if id_match else -1 # id_match.group(1)返回第 1 个括号分组（(\d+)）匹配到的内容

        # 转换设备ID为独热编码
        device_one_hot = torch.zeros(self.num_classes)
        if 0 <= device_id < self.num_classes:
            device_one_hot[device_id] = 1

        # 转换音频为特征向量
        vectors = self.file_to_vectors(file_path)

        # 转换为torch tensor
        features = torch.FloatTensor(vectors)
        features = features.unsqueeze(0)
        label = torch.LongTensor([status])
        device = device_one_hot
        pseudo_label = torch.LongTensor([pseudo_label])
        pseudo_onehot = F.one_hot(pseudo_label, num_classes=self.num_classes).float()
        return features,  pseudo_onehot, device

    def file_to_vectors(self, file_name):
        """
        参考 common.py 中的 file_to_vectors 函数
        将音频文件转换为特征向量

        参数:
        file_name (str): 音频文件路径

        返回:
        numpy.array: 特征向量数组
        """
        # 生成梅尔频谱图
        y, sr = librosa.load(file_name, sr=None, mono=True)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=self.power
        )

        # 转换为对数梅尔能量
        log_mel_spectrogram = 20.0 / self.power * np.log10(
            np.maximum(mel_spectrogram, np.finfo(np.float32).eps)
        )

        # 计算向量总数
        n_vectors = len(log_mel_spectrogram[0, :]) - self.n_frames + 1

        # 跳过太短的音频片段
        if n_vectors < 1:
            return np.empty((0, self.dims))

        # 通过连接多帧生成特征向量
        vectors = np.zeros((n_vectors, self.dims))

        for t in range(self.n_frames):
            vectors[:, self.n_mels * t: self.n_mels * (t + 1)] = \
                log_mel_spectrogram[:, t: t + n_vectors].T

        return vectors








class Wave2Mel(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        # spec =  self.amplitude_to_db(self.mel_transform(x)).squeeze().transpose(-1,-2)
        return self.amplitude_to_db(self.mel_transform(x))


class ASDDataset(Dataset):
    def __init__(self, args, file_list: list, load_in_memory=False):
        self.file_list = file_list
        self.args = args
        self.wav2mel = Wave2Mel(sr=args.sr, power=args.power,
                                n_fft=args.n_fft, n_mels=args.n_mels,
                                win_length=args.win_length, hop_length=args.hop_length)
        self.load_in_memory = load_in_memory



        self.data_list = [self.transform(filename) for filename in file_list] if load_in_memory else []

    def __getitem__(self, item):
        data_item = self.data_list[item] if self.load_in_memory else self.transform(self.file_list[item])
        return data_item

    def transform(self, filename):

        machine = filename.split('/')[-2]

        id_str = re.findall('id_[0-9][0-9]', filename)[0]
        label = self.args.meta2label[machine + '-' + id_str]
        x, _ = librosa.core.load(filename, sr=self.args.sr, mono=True)
        x = x[: self.args.sr * self.args.secs]



        x_wav = torch.from_numpy(x)
        x_mel = self.wav2mel(x_wav)
        return x_wav, x_mel, label



    def __len__(self):
        return len(self.file_list)


class ASDDataset_limit(Dataset):
    def __init__(self, args, file_list: list, load_in_memory=False, max_samples=None):
        self.file_list = file_list
        self.args = args
        self.wav2mel = Wave2Mel(sr=args.sr, power=args.power,
                                n_fft=args.n_fft, n_mels=args.n_mels,
                                win_length=args.win_length, hop_length=args.hop_length)
        self.load_in_memory = load_in_memory

        # 如果指定了最大样本数，则进行采样
        if max_samples is not None and max_samples < len(self.file_list):
            self.file_list = self._sample_balanced_files(max_samples)

        # 初始化数据列表
        self.data_list = [self.transform(filename) for filename in self.file_list] if load_in_memory else []

    def _sample_balanced_files(self, max_samples):
        """
        根据指定的最大样本数，均匀采样各类别的文件

        Args:
            max_samples: 需要采样的最大样本数

        Returns:
            采样后的文件列表
        """
        # 提取所有标签
        labels = []
        for filename in self.file_list:
            machine = filename.split('/')[-2]
            id_str = re.findall('id_[0-9][0-9]', filename)[0]
            label = self.args.meta2label[machine + '-' + id_str]
            labels.append(label)

        # 获取唯一标签
        unique_labels = list(set(labels))
        num_classes = len(unique_labels)

        # 计算每个类别的基本样本数
        samples_per_class = max_samples // num_classes
        remainder = max_samples % num_classes

        # 创建标签到索引的映射
        label_to_indices = {label: [] for label in unique_labels}
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)

        # 从每个类别中采样
        sampled_indices = []
        for i, label in enumerate(unique_labels):
            class_indices = label_to_indices[label]
            # 对于前remainder个类别，多采样一个样本以补足余数
            num_samples = samples_per_class + (1 if i < remainder else 0)
            # 如果类别样本数不足需要的数目，则采样该类别所有样本
            num_samples = min(num_samples, len(class_indices))

            # 随机采样
            import random
            sampled = random.sample(class_indices, num_samples)
            sampled_indices.extend(sampled)

        # 构建采样后的文件列表
        sampled_files = [self.file_list[idx] for idx in sampled_indices]

        return sampled_files

    def __getitem__(self, item):
        data_item = self.data_list[item] if self.load_in_memory else self.transform(self.file_list[item])
        return data_item

    def transform(self, filename):
        machine = filename.split('/')[-2]
        id_str = re.findall('id_[0-9][0-9]', filename)[0]
        label = self.args.meta2label[machine + '-' + id_str]
        x, _ = librosa.core.load(filename, sr=self.args.sr, mono=True)
        x = x[: self.args.sr * self.args.secs]

        x_wav = torch.from_numpy(x)
        x_mel = self.wav2mel(x_wav)
        return x_wav, x_mel, label

    def __len__(self):
        return len(self.file_list)


class ASDDataset_hierarchy(Dataset):
    def __init__(self, args, file_list: list, load_in_memory=False):
        self.file_list = file_list
        self.args = args
        self.wav2mel = Wave2Mel(sr=args.sr, power=args.power,
                                n_fft=args.n_fft, n_mels=args.n_mels,
                                win_length=args.win_length, hop_length=args.hop_length)
        self.load_in_memory = load_in_memory



        self.data_list = [self.transform(filename) for filename in file_list] if load_in_memory else []

    def __getitem__(self, item):
        data_item = self.data_list[item] if self.load_in_memory else self.transform(self.file_list[item])
        return data_item

    def transform(self, filename):

        machine = filename.split('/')[-2]

        id_str = re.findall('id_[0-9][0-9]', filename)[0]
        label = self.args.meta2label[machine + '-' + id_str]
        label = torch.tensor(label)
        x, _ = librosa.core.load(filename, sr=self.args.sr, mono=True)
        x = x[: self.args.sr * self.args.secs]



        x_wav = torch.from_numpy(x)
        x_mel = self.wav2mel(x_wav)
        return x_wav, x_mel, label



    def __len__(self):
        return len(self.file_list)

class ASDDataset_multiview(Dataset):
    def __init__(self, args, file_list: list, load_in_memory=False):
        self.file_list = file_list
        self.args = args
        self.wav2mel = Wave2Mel(sr=args.sr, power=args.power,
                                n_fft=args.n_fft, n_mels=args.n_mels,
                                win_length=args.win_length, hop_length=args.hop_length)
        self.load_in_memory = load_in_memory

        # ==== 新增：从 args 读取噪声增强配置（其余代码不变） ====
        # 是否启用增强（默认 False）
        self.noise_aug_enable = getattr(args, "use_augmentation", False)
        # 添加噪声的概率（默认 0.5）
        self.noise_prob = float(getattr(args, "noise_prob", 0.5))
        # 噪声 SNR 下/上限，单位 dB（默认 10~20dB）
        self.noise_snr_min_db = float(getattr(args, "noise_snr_min_db", 10.0))
        self.noise_snr_max_db = float(getattr(args, "noise_snr_max_db", 20.0))

        self.data_list = [self.transform(filename) for filename in file_list] if load_in_memory else []

    def __getitem__(self, item):
        # 原逻辑：拿到基础三元组（不带增强的 x_mel）
        x_wav, x_mel_base, label = self.data_list[item] if self.load_in_memory else self.transform(self.file_list[item])

        # —— 多视角：仅对 Mel 做两次独立的高斯噪声增强（x_wav 保持不变）——
        x_mel_v1 = self._maybe_add_gaussian_noise_mel(x_mel_base)
        x_mel_v2 = self._maybe_add_gaussian_noise_mel(x_mel_base)

        # 返回四元组：与训练循环中的双视角接口对齐
        return x_wav, x_mel_v1, x_mel_v2, label

        # 如果你仍然需要兼容旧代码（只取单视角），也可以在上面 return 前保留：
        # return (x_wav, x_mel_v1, x_mel_v2, label)

    def transform(self, filename):
        machine = filename.split('/')[-2]
        id_str = re.findall('id_[0-9][0-9]', filename)[0]
        label = self.args.meta2label[machine + '-' + id_str]
        x, _ = librosa.core.load(filename, sr=self.args.sr, mono=True)
        x = x[: self.args.sr * self.args.secs]

        # 随机添加高斯噪声（注：仅作用于波形；若不需要，请保持注释状态）
        # x = self._maybe_add_gaussian_noise(x)

        x_wav = torch.from_numpy(x)
        x_mel = self.wav2mel(x_wav)
        return x_wav, x_mel, label

    def transform_test(self, filename):
        machine = filename.split('/')[-2]
        id_str = re.findall('id_[0-9][0-9]', filename)[0]
        label = self.args.meta2label[machine + '-' + id_str]
        x, _ = librosa.core.load(filename, sr=self.args.sr, mono=True)
        x = x[: self.args.sr * self.args.secs]

        x_wav = torch.from_numpy(x)
        x_mel = self.wav2mel(x_wav)
        return x_wav, x_mel, label

    def __len__(self):
        return len(self.file_list)

    # —— 噪声参数全部由 args 控制：是否启用/概率/SNR范围 ——（保持不删减）
    def _maybe_add_gaussian_noise(self, x: np.ndarray) -> np.ndarray:
        if not self.noise_aug_enable:
            return x
        if random.random() >= self.noise_prob:
            return x
        lo, hi = sorted([self.noise_snr_min_db, self.noise_snr_max_db])
        snr_db = random.uniform(lo, hi)
        sig_power = float(np.mean(x ** 2)) + 1e-12  # 防零
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = sig_power / snr_linear
        noise_std = np.sqrt(noise_power)
        noise = np.random.normal(loc=0.0, scale=noise_std, size=x.shape).astype(x.dtype, copy=False)
        x_noisy = np.clip(x + noise, -1.0, 1.0)
        return x_noisy

    def _maybe_add_gaussian_noise_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """
        对 Mel 频谱做混合增强：白噪声 + SpecAugment（频率遮挡 & 时间遮挡）。
        说明：
          - 仍然沿用 self.noise_aug_enable / self.noise_prob 开关（与原代码一致）
          - 白噪声注入与遮挡在 Mel 空间进行（与原先高斯噪声一致，简单稳定）
          - 形状 [F,T] 或 [1,F,T] 均可，返回形状与输入一致
        """
        if not self.noise_aug_enable:
            return mel
        if random.random() >= self.noise_prob:
            return mel

        # ---- 统一到 [F, T] ----
        squeeze_back = False
        if mel.dim() == 3 and mel.size(0) == 1:
            mel_2d = mel[0]
            squeeze_back = True
        else:
            mel_2d = mel
        device = mel_2d.device
        dtype = mel_2d.dtype
        F, T = mel_2d.shape

        # =========================
        # 1) 白噪声注入（Mel 空间）
        # =========================
        # 内置简单超参（不再从 args 读取，保持“就地可用”）
        p_noise = 0.8
        snr_min_db, snr_max_db = 10.0, 20.0  # 典型稳健区间

        mel_aug = mel_2d
        if random.random() < p_noise:
            # 以 Mel 的均方幅值估计功率
            sig_power = float(torch.mean(mel_aug ** 2).item()) + 1e-12
            snr_db = random.uniform(snr_min_db, snr_max_db)
            snr_linear = 10.0 ** (snr_db / 10.0)
            noise_power = sig_power / snr_linear
            noise_std = float(np.sqrt(noise_power))
            noise = torch.randn_like(mel_aug) * noise_std
            mel_aug = mel_aug + noise

        # =========================
        # 2) SpecAugment 遮挡
        # =========================
        # 频率遮挡（frequency masking）与时间遮挡（time masking）
        # 遮挡值用当前谱的均值，避免在对数域出现极端数值
        p_mask = 1.0
        if random.random() < p_mask:
            mask_val = torch.mean(mel_aug)

            # 频率遮挡
            n_freq_masks = 2
            max_freq_width = max(1, int(0.15 * F))  # 最多遮 15% 频带
            for _ in range(n_freq_masks):
                w = random.randint(1, max_freq_width)
                f0 = random.randint(0, max(0, F - w))
                mel_aug[f0:f0 + w, :] = mask_val

            # 时间遮挡
            n_time_masks = 2
            max_time_width = max(1, int(0.15 * T))  # 最多遮 15% 时间帧
            for _ in range(n_time_masks):
                w = random.randint(1, max_time_width)
                t0 = random.randint(0, max(0, T - w))
                mel_aug[:, t0:t0 + w] = mask_val

        # ---- 还原形状 ----
        if squeeze_back:
            mel_aug = mel_aug.unsqueeze(0)

        return mel_aug.to(device=device, dtype=dtype)


class ASDDataset_random(Dataset):
    def __init__(self, args, file_list: list, load_in_memory=False):
        self.file_list = file_list
        self.args = args
        self.wav2mel = Wave2Mel(sr=args.sr, power=args.power,
                                n_fft=args.n_fft, n_mels=args.n_mels,
                                win_length=args.win_length, hop_length=args.hop_length)
        self.load_in_memory = load_in_memory
        # 添加噪声样本概率参数
        self.noise_prob = 0.1

        # 计算标签的最大值加1作为噪声标签
        self.max_label = max(args.meta2label.values()) if args.meta2label else 0
        self.noise_label = self.max_label + 1

        self.data_list = [self.transform(filename) for filename in file_list] if load_in_memory else []

    def __getitem__(self, item):
        # 以一定概率返回随机噪声样本
        if random.random() < self.noise_prob:
            # 获取正常样本以获取形状信息
            data_item = self.data_list[item] if self.load_in_memory else self.transform(self.file_list[item])
            x_wav_shape = data_item[0].shape
            x_mel_shape = data_item[1].shape

            # 生成与原样本形状相同的随机张量
            noise_wav = torch.randn(x_wav_shape)
            noise_mel = torch.randn(x_mel_shape)

            # 返回噪声样本，标签设为最大标签值加1
            return noise_wav, noise_mel, self.noise_label
        else:
            # 正常返回原始样本
            data_item = self.data_list[item] if self.load_in_memory else self.transform(self.file_list[item])
            return data_item

    def transform(self, filename):
        machine = filename.split('/')[-2]
        id_str = re.findall('id_[0-9][0-9]', filename)[0]
        label = self.args.meta2label[machine + '-' + id_str]
        x, _ = librosa.core.load(filename, sr=self.args.sr, mono=True)
        x = x[: self.args.sr * self.args.secs]

        x_wav = torch.from_numpy(x)
        x_mel = self.wav2mel(x_wav)
        return x_wav, x_mel, label

    def __len__(self):
        return len(self.file_list)

# 没啥用
class ASDDataset_randomlabel(Dataset):
    def __init__(self, args, file_list: list, load_in_memory=False):
        self.file_list = file_list
        self.args = args
        self.wav2mel = Wave2Mel(sr=args.sr, power=args.power,
                                n_fft=args.n_fft, n_mels=args.n_mels,
                                win_length=args.win_length, hop_length=args.hop_length)
        self.load_in_memory = load_in_memory

        # 预设的标签数量
        self.num_labels = 10

        # 为每个文件分配标签
        self.labels = self._assign_labels()

        self.data_list = [self.transform(filename) for filename in
                          self.file_list] if load_in_memory else []

    def _assign_labels(self):
        """
        根据预设的标签数量将文件列表划分并分配标签
        """
        total_files = len(self.file_list)
        labels = []

        # 计算每份的基本大小和余数
        base_size = total_files // self.num_labels
        remainder = total_files % self.num_labels

        file_idx = 0
        for label in range(self.num_labels):
            # 前remainder个标签多分配一个文件
            current_size = base_size + 1 if label < remainder else base_size

            # 为当前标签分配文件
            for _ in range(current_size):
                labels.append(label)
                file_idx += 1

        return labels

    def __getitem__(self, item):
        if self.load_in_memory:
            data_item = self.data_list[item]
        else:
            data_item = self.transform(self.file_list[item])
        # 添加标签到返回值中
        x_wav, x_mel = data_item
        label = self.labels[item]
        return x_wav, x_mel, label

    def transform(self, filename):
        machine = filename.split('/')[-2]
        # 不再使用原有的标签获取方式
        # id_str = re.findall('id_[0-9][0-9]', filename)[0]
        # label = self.args.meta2label[machine + '-' + id_str]

        x, _ = librosa.core.load(filename, sr=self.args.sr, mono=True)
        x = x[: self.args.sr * self.args.secs]

        x_wav = torch.from_numpy(x)
        x_mel = self.wav2mel(x_wav)
        return x_wav, x_mel

    def __len__(self):
        return len(self.file_list)

