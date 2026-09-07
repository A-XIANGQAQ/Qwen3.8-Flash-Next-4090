# 运维手册 (Operations)

## 1. 启动 (Start)

```bash
bash scripts/start_lsglang.sh
# 或长 SSH 会话用 nohup（脚本内已 nohup）
tail -f /tmp/lsglang.log
```

- 就绪标志：`The server is fired up and ready to roll!`（首次 ~5 分钟）
- 期间 `/health` 可能短暂 503，属正常

## 2. 观察 (Observe)

```bash
curl -s localhost:8000/health
watch -n 1 nvidia-smi
free -h                     # 模型 126GB resident + 运行时 ≈ 200GB
tail -f /tmp/lsglang.log    # 关注: Traceback / OOM / Xid / swap 增长
```

已知非致命日志：sarashina2_vision import error、mrope key warning、Data race warning——均无害。

## 3. 停止 / 重启 (Stop / Restart)

```bash
# 优雅停（等运行中请求完成，长请求可能很久——sglang 会打印 Gracefully exiting）
pkill -f 'sglang serve'
# 立即停（中断请求）
pkill -9 -f 'sglang serve'
# 重启
bash scripts/start_lsglang.sh
```

⚠️ 不要用 `pgrep -f 'sglang serve'` 取 PID 后再 kill——该模式会匹配到命令行含此字符串的 shell 自身。

## 4. 日志 (Logs)

- `/tmp/lsglang.log`（启动脚本指定）
- 关键日志行：
  - `Load weight end. elapsed=...`——加载完成
  - `KV Cache is allocated. #tokens: ...`——KV 池大小（256K 可用需 ≥262144）
  - `Mamba Cache is allocated.`——Mamba/SSM 池
  - `gen throughput (token/s)`——**服务内部 decode 吞吐**（仅作参考，见 TROUBLESHOOTING）
  - `accept rate`——MTP 开启时的 draft 接受率

## 5. 常用 curl 验证 (Sanity Checks)

```bash
# 模型信息（工具客户端读这个限制长度）
curl -s localhost:8000/v1/models

# 中文请求（关思考）
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<served_name>","messages":[{"role":"user","content":"你好"}],"max_tokens":64,"stream":false,"chat_template_kwargs":{"enable_thinking":false}}'

# 工具调用（必须 --tool-call-parser qwen3_coder 才能解析出结构化 tool_calls）
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<served_name>","messages":[{"role":"user","content":"北京天气？"}],"max_tokens":128,"tools":[{"type":"function","function":{"name":"get_weather","description":"查天气","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'
# 期望 message.tool_calls 为结构化数组（非 null）
```

## 6. 边界行为 (Boundary Behavior)

- 输入 > 262144 tokens → HTTP 400（context-length 上限）
- prompt + max_tokens > 265216 → 400（准入）
- 生成长度到 262144 总长 → 截断（finish_reason=length）
- 多客户端并发：`--max-running-requests 2`，超出排队

## 7. GUI / 无头主机共存 (Headless Notes)

- 128GB 权重 + 运行时需 ~200GB RAM——若主机有桌面环境，建议服务运行期间关闭 GUI（LightDM 等）
- 远程访问：SSH local forwarding 或 loopback + 反向代理（nginx 需配 `proxy_read_timeout`，默认 60s 会掐断长请求 → 502；建议 900s）
