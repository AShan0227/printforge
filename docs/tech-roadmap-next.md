# 技术路线图 — 下一代输入方式（Feynman-X）

## v2.0 — 多输入源融合

### 输入方式扩展

| 输入 | 技术 | 优先级 | 描述 |
|------|------|--------|------|
| 单张照片 | ✅ 已有 | — | Hunyuan3D-2 |
| 多张照片 | ✅ 已有 | — | multi_view.py |
| 文字描述 | ✅ 已有 | — | text_to_3d.py |
| **手机 3D 扫描** | 3D Gaussian Splatting → Mesh | P1 | iPhone LiDAR / Android depth sensor |
| **视频环绕拍摄** | Multi-view from video frames | P1 | 手机绕物体拍一圈 |
| **草图手绘** | Sketch-to-3D diffusion | P2 | 铅笔画 → 3D |
| **语音描述** | Speech→Text→3D | P3 | "给我一个圆角方形花瓶" |

### 3D Gaussian Splatting → Print 路线

**突破性发现：** 2025 年 3DGS 已经可以转 mesh 了

```
手机拍视频（绕物体一圈）
    ↓ 抽帧
多张图片 (30-60 张)
    ↓ 3D Gaussian Splatting (COLMAP + 3DGS)
高斯点云
    ↓ SuGaR / Poisson Reconstruction
Mesh
    ↓ PrintForge watertight pipeline
3MF → 打印
```

**开源工具链：**
- COLMAP（Structure from Motion）
- 3DGS（原始论文实现）
- SuGaR（Gaussian→Mesh，开源）
- PrintForge（Mesh→Print-ready）

**类比（Feynman-X）：** 就像手机相册的"回忆视频"功能——拍几张照片自动生成一段视频。我们做的是：拍几张照片自动生成一个可打印的实物。

### 应用场景延伸
- **逆向工程**：扫描一个零件 → 3D 打印备件
- **纪念品**：扫描家里的物品 → 缩小版 3D 打印纪念
- **电商**：360° 产品扫描 → 3D 打印样品寄给客户
