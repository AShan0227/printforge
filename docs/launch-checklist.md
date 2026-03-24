# 发布检查清单（Munger 审查）

## 必须在发布前完成

- [x] E2E 验证：Photo → STL → Bambu Studio 切片成功
- [x] Watertight 100% 保证
- [x] 91+ 测试通过
- [x] README 完整
- [x] LICENSE (MIT)
- [x] Docker 支持
- [x] CI (GitHub Actions)
- [ ] **用户协议/TOS** — 正在开发
- [ ] **至少 3 个示例模型** 上传到 MakerWorld/Printables
- [ ] **演示 GIF/视频** 嵌入 README
- [x] 竞品对比数据
- [x] 定价策略

## 发布后 48 小时监控

- [ ] Reddit 帖子互动率
- [ ] GitHub Stars 增长
- [ ] Issues 响应（<2h）
- [ ] HF Space 稳定性
- [ ] 用户报告的第一个 bug

## 回滚计划
- 如果 HF Space 挂了 → README 加 "需要本地 GPU" 说明
- 如果用户报告切片失败 → 立即修复并发 hotfix
- 如果出现版权投诉 → 添加 DMCA 流程 + 内容检测
