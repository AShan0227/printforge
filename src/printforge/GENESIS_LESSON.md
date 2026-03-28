# Genesis 教训

## 2026-03-28 根因
file_write 不展开 ~ 导致文件写到错误位置。已修复。

## 验证链
1. file_write 写入
2. shell_exec ls -la 确认存在
3. shell_exec cat 确认内容
