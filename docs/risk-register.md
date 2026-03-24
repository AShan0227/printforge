# 风险登记册（Munger 持续更新）

## 🔴 高风险

### R1: HuggingFace Space 依赖
**风险：** Tencent 随时可能关闭/限流 Hunyuan3D-2 Space
**影响：** 核心推理完全失效
**缓解：** 
- 已实现 5 级 fallback（Hunyuan→mini→TripoSR→local→placeholder）
- TODO：下载 Hunyuan3D-2 模型权重到本地（~2GB），实现真正的离线推理
**状态：** ⚠️ 部分缓解

### R2: 版权/肖像权
**风险：** 用户上传明星照片/品牌 logo/受版权保护的图片生成 3D 模型
**影响：** 法律诉讼，产品下架
**缓解：**
- TODO：添加用户协议——用户对上传内容负责
- TODO：添加 NSFW/版权检测（可选开关）
- MIT 开源 = 用户自担风险（但不能免除平台责任如果做 SaaS）
**状态：** ❌ 未缓解

### R3: 模型质量不稳定
**风险：** 复杂物体/有机形状生成质量差，用户期望与实际不符
**影响：** 差评，用户流失
**缓解：**
- 已有 QualityScorer（开发中）预警低质量输出
- TODO：在 Web UI 显示质量评分和建议
- TODO："不满意免费重试"策略
**状态：** ⚠️ 部分缓解

## 🟡 中风险

### R4: 竞品价格战
**风险：** Meshy 推出免费版或大幅降价
**缓解：** 我们的差异化是开源+本地+watertight 保证，不在价格上竞争

### R5: Bambu Studio 格式变更
**风险：** Bambu Studio 更新后我们的 3MF 不兼容
**缓解：** 跟踪 Bambu Studio 更新日志，持续验证

### R6: GPU 推理成本
**风险：** 如果从免费 Space 切到自建 GPU，成本骤增
**缓解：** Dalio 的模型显示 92 用户即可盈亏平衡

## 🟢 已缓解

### R7: Watertight 失败 → ✅ 已修复
voxelize→fill→marching cubes 100% 保证

### R8: 推理速度太慢 → ✅ 已优化
缓存层避免重复推理，42s 首次 / 0s 缓存命中
