### Retrieval-based-Voice-Conversion-WebUI


 ✅ 原有基础功能

 ✨ 核心改动

- 自动切换环境变量：自动区分CUDA/ROCm环境，设置对应显卡可见性变量（ROCR_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES）。

- 设备逻辑统一：新增全局device变量，将分散的设备迁移代码统一为.to(device)，提升多设备环境下的稳定性。

- TQDM进度条：集成tqdm库，训练时显示实时进度条，包含epoch进度、batch计数、动态列宽，直观掌握训练节奏。

- 动态日志间隔：根据每个epoch的batch总数自适应调整日志打印频率，默认按batch总数5等分设置间隔。

- 可配置进度条后缀：通过环境变量RVC_SHOW_POSTFIX控制，支持在进度条实时显示生成器损失、判别器损失、Mel损失等指标。

- 动态梯度裁剪：通过环境变量USE_AGC控制，防止梯度爆炸

- GPU缓存重建策略：修改GPU数据缓存逻辑，每个epoch强制清空并重建缓存。

---

## 🎛️ 高级训练配置（环境变量）

通过环境变量微调训练行为。

### `RVC_SHOW_POSTFIX` - 实时损失显示
**用途**：控制是否在训练进度条（tqdm）中实时显示当前损失值。

| 设置值 | 效果 | 适用场景 |
|--------|------|----------|
| `1`, `true`, `yes` | 在进度条右侧显示 `g/d/mel/kl` 实时损失 | **推荐**：监控训练动态 |
| `0` (默认) | 仅显示进度条，不显示损失数值 | 减少日志干扰，提高IO性能 |

**使用示例**：
```bash
# Linux/macOS
export RVC_SHOW_POSTFIX=1
```

---

### `USE_AGC` - 自适应梯度裁剪 
**用途**：根据判别器与生成器的梯度比例动态调整裁剪阈值

| 设置值 | 效果 | 适用场景 |
|--------|------|----------|
| `1`, `true`, `yes` | 启用AGC，防止梯度爆炸，自动平衡 D/G 训练 | 推荐用于小批次或不稳定训练 |
| `0` (默认) | 使用硬裁剪（clip_grad_value_） | 标准训练环境，或需要固定梯度范围时 |

**AGC 特性说明**：
- **自适应阈值**：根据历史梯度分布的85百分位数动态调整（范围：20.0 ~ 120.0）
- **耦合比例**：维持 G/D 梯度范数比例为 2:1，防止判别器过度强势
- **快速响应**：30步历史窗口，3步预热，适合快速变化的GAN动态

**使用示例**：
```bash
# 启用AGC（推荐用于小批次或不稳定训练）
export USE_AGC=1
```

**TensorBoard 监控指标**（启用后新增）：
- `agc/clip_norm_d` & `agc/clip_norm_g`: 实际裁剪后的梯度范数
- `agc/clip_ratio_d` & `agc/clip_ratio_g`: 裁剪比例（0~1，越接近1说明裁剪越激进）
- `agc/threshold_d` & `agc/threshold_g`: 当前动态阈值
- `agc/gd_ratio`: 生成器与判别器阈值比例（目标为2.0）

**⚠️ 注意事项**：
1. **ROCm用户**：建议开启，代码已针对ROCm/HIP后端优化AGC参数
2. **显存影响**：AGC计算需要额外的显存开销（约100-200MB），若显存紧张可关闭


### ROCm性能
PyTorch	2.9.1+rocm6.4	适配 ROCm 6.4 + RX 9070 XT

启用config中的fp16_run，平均每30秒左右即可完成一轮训练

<img width="765" height="340" alt="image" src="https://github.com/user-attachments/assets/faf5077d-a54f-4d2b-806b-115d1649dd74" />



环境配置和使用请参阅[原RVC项目](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

致谢

本项目基于原RVC项目修改为适配我自己rocm环境的版本，感谢原作者及社区贡献者的开源成果。

