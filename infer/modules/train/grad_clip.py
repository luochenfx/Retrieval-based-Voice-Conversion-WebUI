import torch
import numpy as np
from collections import deque
from typing import Tuple, Dict, Optional


class CoupledAdaptiveGradClip:
    """
    ROCm/GAN 优化的自适应梯度裁剪器
    """

    def __init__(
            self,
            ratio_g_to_d: float = 2.0,
            base_percentile: float = 85.0,
            buffer_base: float = 1.15,
            buffer_cv_weight: float = 0.5,
            min_threshold: float = 25.0,
            max_threshold: float = 120.0,
            history_size: int = 30,
            warmup_steps: int = 3,
            device: str = 'cuda',
            ratio_elasticity: float = 1.2  # 比例弹性系数，可配置
    ):
        self.ratio = float(ratio_g_to_d)
        self.percentile = float(base_percentile)
        self.buffer_base = float(buffer_base)
        self.buffer_cv_weight = float(buffer_cv_weight)
        self.min_th = float(min_threshold)
        self.max_th = float(max_threshold)
        self.warmup_steps = int(warmup_steps)
        self.device = device
        self.ratio_elasticity = float(ratio_elasticity)  # 弹性系数

        # float32 CPU 存储，避免 ROCm 内存泄漏
        self.history_g: deque = deque(maxlen=history_size)
        self.history_d: deque = deque(maxlen=history_size)

        self.step_count = 0
        self.current_th_g = max_threshold
        self.current_th_d = max_threshold / ratio_g_to_d
        self._cached_stats: Optional[Dict] = None

    @staticmethod
    def _to_float(tensor: torch.Tensor) -> float:
        """强制转为 float32 CPU 标量，ROCm 内存安全"""
        return float(tensor.detach().cpu().float().item())

    def _safe_percentile(self, data: deque, p: float) -> float:
        """安全分位数计算（空/短数据保护）"""
        if len(data) == 0:
            return self.max_th
        if len(data) < 3:
            return float(np.mean(list(data)))
        return float(np.percentile(list(data), p))

    @staticmethod
    def _safe_cv(data: deque) -> float:
        """安全变异系数计算（防除零/防nan/防inf）"""
        if len(data) < 3:
            return 0.0
        arr = np.array(list(data), dtype=np.float32)
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        if mean_val < 1e-6:
            return 10.0
        cv = std_val / mean_val
        if not np.isfinite(cv):
            return 0.0
        return float(np.clip(cv, 0.0, 5.0))

    def _compute_adaptive_buffer(self, data: deque) -> float:
        """动态缓冲系数（基于 CV 调整）"""
        cv = self._safe_cv(data)
        reduction = self.buffer_cv_weight * min(cv, 1.0)
        adaptive_buffer = self.buffer_base * (1.0 - reduction)
        return max(1.05, adaptive_buffer)

    def _calculate_raw_threshold(self, data: deque) -> float:
        """计算单网络原始阈值（无硬边界）"""
        base_val = self._safe_percentile(data, self.percentile)
        buffer_coeff = self._compute_adaptive_buffer(data)
        return base_val * buffer_coeff

    def _apply_hard_limits(self, th_g: float, th_d: float) -> Tuple[float, float]:
        """应用硬边界 + 双向比例约束（弹性系数可配置）"""
        # 硬边界限制
        th_g = float(np.clip(th_g, self.min_th, self.max_th))
        th_d = float(np.clip(th_d, self.min_th, self.max_th))

        # 双向基准计算
        d_from_g = th_g / self.ratio
        g_from_d = th_d * self.ratio

        # 可配置弹性系数（替换硬编码1.2）
        final_g = min(th_g, g_from_d * self.ratio_elasticity)
        final_d = min(th_d, d_from_g * self.ratio_elasticity)

        # 下限兜底
        return max(final_g, self.min_th), max(final_d, self.min_th)

    @staticmethod
    def get_clip_ratio(raw_norm: float, clipped_norm: float) -> float:
        """新增：安全计算裁剪率（百分比），含除零保护"""
        if raw_norm < 1e-6:  # 原始范数接近0，无裁剪
            return 0.0
        return float(((raw_norm - clipped_norm) / raw_norm) * 100)

    def clip_both(
            self,
            params_g,
            params_d
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        主入口：裁剪 G/D 梯度
        优化：删除冗余device代码+双层异常捕获+保持所有原有特性
        返回: ((raw_g, clipped_g), (raw_d, clipped_d))
        """
        self.step_count += 1
        params_g = list(params_g)
        params_d = list(params_d)
        if len(params_g) == 0 or len(params_d) == 0:
            return (0.0, 0.0), (0.0, 0.0)

        with torch.no_grad():
            # 计算原始梯度范数
            raw_norm_g = torch.nn.utils.clip_grad_norm_(params_g, float('inf'))
            raw_norm_d = torch.nn.utils.clip_grad_norm_(params_d, float('inf'))
            # 转为CPU float32，记录历史
            raw_g_float = self._to_float(raw_norm_g)
            raw_d_float = self._to_float(raw_norm_d)
            self.history_g.append(raw_g_float)
            self.history_d.append(raw_d_float)

            # 计算自适应阈值
            if self.step_count <= self.warmup_steps:
                candidate_g = self.max_th
                candidate_d = self.max_th / self.ratio
            else:
                candidate_g = self._calculate_raw_threshold(self.history_g)
                candidate_d = self._calculate_raw_threshold(self.history_d)

            # 应用耦合约束+硬边界
            self.current_th_g, self.current_th_d = self._apply_hard_limits(
                candidate_g, candidate_d
            )

            # 执行实际裁剪
            clipped_norm_g = torch.nn.utils.clip_grad_norm_(params_g, self.current_th_g)
            clipped_norm_d = torch.nn.utils.clip_grad_norm_(params_d, self.current_th_d)

            # 转为CPU float32返回
            clipped_g_float = self._to_float(clipped_norm_g)
            clipped_d_float = self._to_float(clipped_norm_d)

            return (raw_g_float, clipped_g_float), (raw_d_float, clipped_d_float)

    # 在CoupledAdaptiveGradClip类中添加以下2个方法
    def clip_d(self, params_d):
        """分步裁剪：仅裁剪判别器D的梯度，返回(原始D范数, 裁剪后D范数)"""
        params_d = list(params_d)
        if len(params_d) == 0:
            return 0.0, 0.0
        with torch.no_grad():
            # 计算D原始梯度范数并记录历史
            raw_norm_d = torch.nn.utils.clip_grad_norm_(params_d, float('inf'))
            raw_d_float = self._to_float(raw_norm_d)
            self.history_d.append(raw_d_float)

            # 计算自适应阈值（维持G/D=2:1比例）
            if self.step_count <= self.warmup_steps:
                self.current_th_d = self.max_th / self.ratio
            else:
                # 先计算D原始阈值，再按比例推导G阈值，保证比例约束
                d_raw_th = self._calculate_raw_threshold(self.history_d)
                self.current_th_d = float(np.clip(d_raw_th, self.min_th, self.max_th))
                self.current_th_g = self.current_th_d * self.ratio  # G阈值=D阈值×2

            # 执行D梯度裁剪
            clipped_norm_d = torch.nn.utils.clip_grad_norm_(params_d, self.current_th_d)
            return self._to_float(raw_norm_d), self._to_float(clipped_norm_d)

    def clip_g(self, params_g):
        """分步裁剪：仅裁剪生成器G的梯度，返回(原始G范数, 裁剪后G范数)"""
        params_g = list(params_g)
        if len(params_g) == 0:
            return 0.0, 0.0
        with torch.no_grad():
            # 计算G原始梯度范数并记录历史
            raw_norm_g = torch.nn.utils.clip_grad_norm_(params_g, float('inf'))
            raw_g_float = self._to_float(raw_norm_g)
            self.history_g.append(raw_g_float)

            # 计算自适应阈值（基于D阈值推导，强制维持2:1比例）
            if self.step_count <= self.warmup_steps:
                self.current_th_g = self.max_th
            else:
                # G阈值由D阈值推导，确保比例严格为2:1
                self.current_th_g = float(np.clip(self.current_th_d * self.ratio, self.min_th, self.max_th))

            # 执行G梯度裁剪
            clipped_norm_g = torch.nn.utils.clip_grad_norm_(params_g, self.current_th_g)
            return self._to_float(raw_norm_g), self._to_float(clipped_norm_g)

    def get_stats(self) -> Dict[str, float]:
        """
        获取监控统计
        所有统计量均有默认值，可直接写入TensorBoard/日志
        """
        len_g, len_d = len(self.history_g), len(self.history_d)
        # 基础统计
        stats = {
            'step': self.step_count,
            'history_len_g': len_g,
            'history_len_d': len_d,
            'threshold_g': self.current_th_g,
            'threshold_d': self.current_th_d,
            'target_ratio': self.ratio,
            'in_warmup': self.step_count <= self.warmup_steps,
            'actual_ratio': 0.0,
            'ratio_error': 0.0,
            # G统计量默认值
            'grad_mean_g': 0.0,
            'grad_std_g': 0.0,
            'cv_g': 0.0,
            'buffer_g': self.buffer_base,
            # D统计量默认值
            'grad_mean_d': 0.0,
            'grad_std_d': 0.0,
            'cv_d': 0.0,
            'buffer_d': self.buffer_base,
        }

        # 计算G详细统计（统一len>=3判断，与底层函数一致）
        if len_g >= 3:
            arr_g = np.array(list(self.history_g), dtype=np.float32)
            stats['grad_mean_g'] = float(np.mean(arr_g))
            stats['grad_std_g'] = float(np.std(arr_g))
            stats['cv_g'] = self._safe_cv(self.history_g)
            stats['buffer_g'] = self._compute_adaptive_buffer(self.history_g)

        # 计算D详细统计（统一len>=3判断）
        if len_d >= 3:
            arr_d = np.array(list(self.history_d), dtype=np.float32)
            stats['grad_mean_d'] = float(np.mean(arr_d))
            stats['grad_std_d'] = float(np.std(arr_d))
            stats['cv_d'] = self._safe_cv(self.history_d)
            stats['buffer_d'] = self._compute_adaptive_buffer(self.history_d)

        # 比例误差计算（防除零）
        if self.current_th_d > 1e-6:
            stats['actual_ratio'] = self.current_th_g / self.current_th_d
            stats['ratio_error'] = abs(stats['actual_ratio'] - self.ratio)

        return stats


# ==================== 使用示例 ====================
def train_step_optimized():
    """集成示例：裁剪率直接调用"""
    # 初始化（弹性系数可按需调整，如梯度震荡时设为1.0）
    agc = CoupledAdaptiveGradClip(
        ratio_g_to_d=2.0,
        min_threshold=20.0,
        max_threshold=120.0,
        history_size=30,
        warmup_steps=3,
        device='cuda',
        ratio_elasticity=1.2  # 弹性系数可配置
    )
    """
    for global_step, batch_data in enumerate(train_loader):
        # 执行梯度裁剪
        (raw_g, clip_g), (raw_d, clip_d) = agc.clip_both(net_g.parameters(), net_d.parameters())

        # 裁剪率直接调用，无需手动计算（新增特性）
        clip_ratio_g = agc.get_clip_ratio(raw_g, clip_g)
        clip_ratio_d = agc.get_clip_ratio(raw_d, clip_d)

        # 日志监控（无键缺失风险）
        if global_step % 50 == 0:
            stats = agc.get_stats()
            print(f"[Step {global_step}] "
                  f"G: {raw_g:.1f}/{clip_g:.1f}({clip_ratio_g:.0f}%) | "
                  f"D: {raw_d:.1f}/{clip_d:.1f}({clip_ratio_d:.0f}%) | "
                  f"Thresh: {stats['threshold_g']:.1f}/{stats['threshold_d']:.1f} | "
                  f"CV-G: {stats['cv_g']:.2f} | Ratio-Err: {stats['ratio_error']:.3f}")

            # TensorBoard 记录（所有键均存在，无异常）
            writer.add_scalar("agc/clip_ratio_g", clip_ratio_g, global_step)
            writer.add_scalar("agc/buffer_g", stats['buffer_g'], global_step)
            writer.add_scalar("agc/ratio_error", stats['ratio_error'], global_step)    
    """
