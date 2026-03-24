# PrintForge 商业化全景规划

## 市场数据

- 3D 打印软件市场：2025 年 $30 亿，2030 年 $73 亿（CAGR 19%）
- 3D 打印切片器市场：2025 年 $30.7 亿，2029 年 $81.6 亿（CAGR 27.7%）
- 拓竹 2024 年收入 $7.6-8.3 亿，120 万台出货，29% 全球份额
- Thingiverse 年收入 ~$1000-1500 万（广告模式）
- MakerWorld（拓竹模型平台）快速增长，创作者激励模式

## 产品全景（不只是 2D→3D）

```
PrintForge 产品矩阵
│
├── 🎨 创建层 (Input → 3D Model)
│   ├── Image to 3D     ← v0.1 已有
│   ├── Text to 3D      ← 用 LLM 理解描述 → 生成模型
│   ├── Sketch to 3D    ← 手绘草图 → 3D 模型
│   ├── Multi-view to 3D ← 多张照片 → 高精度重建
│   └── 3D Scan to Print ← 手机 LiDAR → 扫描 → 优化 → 打印
│
├── ✏️ 编辑层 (Edit & Customize)
│   ├── AI Mesh Editor    ← "把手柄加长 2cm" 自然语言编辑
│   ├── Part Splitter     ← 大模型自动拆件（超出打印机尺寸）
│   ├── Texture Painter   ← AI 自动上色（多色打印机）
│   ├── Parametric Resize ← 智能缩放（保持结构强度）
│   └── Assembly Designer ← 多部件卡扣/螺丝孔自动生成
│
├── 🔧 优化层 (Print-Ready)
│   ├── Watertight Fix    ← v0.1 已有（SDF→Marching Cubes）
│   ├── Support Optimizer ← AI 推荐最优打印方向和支撑
│   ├── Strength Analysis ← FEA 简化版，标记薄弱点
│   ├── Material Advisor  ← 根据用途推荐材料（PLA/PETG/ABS/TPU）
│   └── Cost Estimator    ← 材料用量 + 打印时间 + 电费估算
│
├── 🖨️ 打印层 (Print & Monitor)
│   ├── Bambu Studio 直连 ← API 直接发送打印（v0.2）
│   ├── Print Queue       ← 批量打印管理
│   ├── Remote Monitor    ← 打印进度 + 异常检测
│   └── Multi-Printer     ← 多台打印机协同
│
├── 🏪 社区层 (Share & Monetize)
│   ├── Model Marketplace ← 用户分享/出售模型
│   ├── Print Service     ← 没有打印机的用户 → 下单给有打印机的人
│   ├── Template Library  ← 可参数化的模型模板（手机壳/花瓶/齿轮...）
│   └── Creator Program   ← 设计师激励（类似 MakerWorld）
│
└── 🧠 AI 层 (Intelligence)
    ├── Print Failure Prediction ← 预测打印失败并建议修复
    ├── Quality Assessment       ← 照片评估打印质量
    ├── Design Suggestion        ← "这个设计建议加肋骨增强"
    └── Learning from Failures   ← 社区打印失败数据 → 改进模型
```

## 核心能力矩阵

### P0 — MVP（现在 - 4 周）
| 功能 | 状态 | 描述 |
|------|------|------|
| Image to 3D | ✅ v0.1 | TripoSR → SDF → Marching Cubes → 3MF |
| Watertight Fix | ✅ v0.1 | SDF 强制 watertight |
| CLI | ✅ v0.1 | `printforge photo.jpg -o model.3mf` |
| Web API | ✅ v0.1 | FastAPI `/api/generate` |
| Web UI | 🔜 Week 2 | 上传 → 预览 → 下载 |
| Multi-view Enhancement | 🔜 Week 3 | Zero123++ 多视角提升背面质量 |

### P1 — 产品化（1-2 月）
| 功能 | 描述 | 商业价值 |
|------|------|---------|
| Text to 3D | "一个猫形花瓶" → 3D 模型 | 降低创作门槛到零 |
| Part Splitter | 自动拆件打印 | 解决"模型太大打不下"痛点 |
| Support Optimizer | AI 推荐最优打印方向 | 减少支撑浪费 50%+ |
| Bambu Studio 直连 | 一键发送打印 | 从模型到实物零操作 |
| Material Advisor | 推荐 PLA/PETG/ABS | 新手不知道用什么材料 |

### P2 — 生态化（3-6 月）
| 功能 | 描述 | 商业价值 |
|------|------|---------|
| Model Marketplace | 用户分享/出售模型 | 平台抽成，类似 MakerWorld |
| Template Library | 参数化模板 | 手机壳/齿轮/挂钩等定制 |
| Print Service | 没打印机 → 下单 | 连接供需，佣金模式 |
| Sketch to 3D | 手绘草图转 3D | 设计师/学生刚需 |
| Strength Analysis | 简化 FEA | 功能件必须知道哪里会断 |

### P3 — 智能化（6-12 月）
| 功能 | 描述 | 商业价值 |
|------|------|---------|
| Failure Prediction | 预测打印失败 | 节省时间和材料 |
| AI Mesh Editor | 自然语言编辑模型 | "把这个角磨圆" |
| Assembly Designer | 自动生成卡扣/螺丝孔 | 多部件组装必备 |
| Learning Pipeline | 社区失败数据 → 改进 | 数据飞轮，越用越好 |

## 商业模式

### 收入来源

| 来源 | 模式 | 预估 |
|------|------|------|
| **Freemium 订阅** | 免费 5 次/天，Pro $9.9/月无限制 | 主要收入 |
| **Marketplace 抽成** | 模型交易 15% 佣金 | 生态收入 |
| **Print Service 佣金** | 打印订单 10% 佣金 | 连接收入 |
| **企业 API** | 按量计费，嵌入到其他产品 | B2B 收入 |
| **Template 付费** | 高级参数化模板 $1-$5/个 | 内容收入 |

### 定价

| Plan | 价格 | 功能 |
|------|------|------|
| Free | $0 | 5 次/天，Image to 3D，STL 输出 |
| Pro | $9.9/月 | 无限次，3MF，Text to 3D，Part Splitter，Priority |
| Team | $29.9/月 | Multi-printer，Print Queue，API access |
| Enterprise | 联系我们 | 自部署，定制模型，SLA |

## 竞争差异化

| | PrintForge | Meshy | Sloyd | 3D AI Studio |
|---|---|---|---|---|
| **开源** | ✅ MIT | ❌ | ❌ | ❌ |
| **本地运行** | ✅ | ❌ 云端 | ❌ 云端 | ❌ 云端 |
| **打印专用** | ✅ watertight 保证 | ⚠️ 通用 | ⚠️ 通用 | ⚠️ 通用 |
| **拓竹集成** | ✅ 直连 | ❌ | ❌ | ❌ |
| **多部件** | ✅ (P1) | ❌ | ❌ | ❌ |
| **Text to 3D** | ✅ (P1) | ✅ | ✅ | ✅ |
| **隐私** | ✅ 本地推理 | ❌ 上传云端 | ❌ | ❌ |
| **价格** | 免费+$9.9 | $10/月 | 免费+ | $9.99/月 |
