### Retrieval-based-Voice-Conversion-WebUI


 ✅ 原有基础功能

 ✨ 核心改动

- 自动切换环境变量：自动区分CUDA/ROCm环境，设置对应显卡可见性变量（ROCR_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES）。

- 设备逻辑统一：新增全局device变量，将分散的设备迁移代码统一为.to(device)，提升多设备环境下的稳定性。

- TQDM进度条：集成tqdm库，训练时显示实时进度条，包含epoch进度、batch计数、动态列宽，直观掌握训练节奏。

- 动态日志间隔：根据每个epoch的batch总数自适应调整日志打印频率，默认按batch总数5等分设置间隔。

- 可配置进度条后缀：通过环境变量RVC_SHOW_POSTFIX控制，支持在进度条实时显示生成器损失、判别器损失、Mel损失等指标。

- GPU缓存重建策略：修改GPU数据缓存逻辑，每个epoch强制清空并重建缓存。


环境配置和使用请参阅[原RVC项目](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

致谢

本项目基于原RVC项目开发，感谢原作者及社区贡献者的开源成果。

