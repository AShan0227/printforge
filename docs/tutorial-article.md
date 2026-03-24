# 我用 AI 把一张照片变成了 3D 打印品

> 42 秒，一张照片，一个可以摸到的实物。

## 问题

我有一台拓竹 A1 打印机，但我不会 3D 建模。

每次想打印一个东西，我要么在 Thingiverse 上找别人做好的模型（经常找不到我想要的），要么花几个小时学 Blender（然后放弃）。

我想要的很简单：**拍张照片，打印出来。**

## 解决

我写了 PrintForge——一个开源的命令行工具。

```bash
pip install printforge
printforge image my_cat.jpg -o cat_figurine.3mf
```

42 秒后，你得到一个 3MF 文件。直接拖进 Bambu Studio，切片，打印。

## 它是怎么工作的

1. **AI 看图** — Hunyuan3D-2（腾讯开源的 AI）分析照片，推断 3D 形状
2. **变防水** — 通过体素化 + Marching Cubes 算法，生成的模型 100% 密封（这是 3D 打印的基本要求）
3. **打印优化** — 自动缩放到你指定的大小，检查壁厚，简化面数

整个过程不需要 GPU，不需要上传到云端，不需要注册账号。

## 实测

我用一张简单的方块图片测试了完整流程：

- **推理时间：** 42 秒
- **输出：** 166,870 个三角面，watertight = True
- **Bambu Studio 切片：** 成功 ✅
- **预估打印时间：** 1 小时 6 分钟
- **预估用料：** 23.84g PLA

## 和 Meshy 比有什么不同？

| | PrintForge | Meshy |
|---|---|---|
| 开源 | ✅ | ❌ |
| 本地运行 | ✅ | ❌ (云端) |
| 隐私 | ✅ 照片不上传 | ❌ 上传到 Meshy 服务器 |
| 切片成功率 | **100%** | 97% |
| 价格 | 免费 | $20/月 |

## 局限

说实话，AI 生成的 3D 模型目前有几个问题：

1. **背面靠猜** — 一张照片只有正面信息，AI 猜测背面。复杂物体的背面可能不准
2. **有机形状差** — 人脸、动物等有机形状精度有限。硬表面物体（杯子、logo、玩具）效果最好
3. **细节丢失** — 小于 0.4mm 的细节会被抹掉（这也是 FDM 打印的物理限制）

## 试试看

```bash
pip install printforge
printforge image your_photo.jpg -o model.3mf
```

或者用 Web 界面：
```bash
printforge serve
# 打开 http://localhost:8000
```

GitHub: https://github.com/AShan0227/printforge

MIT 开源，欢迎贡献。
