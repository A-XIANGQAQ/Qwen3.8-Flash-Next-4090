# Vendored benchmark: vLLM v0.7.3 + local patches

这些文件来自 **[vllm-project/vllm](https://github.com/vllm-project/vllm) `v0.7.3`** 的 `benchmarks/`：

| 文件 | 上游 sha256 |
|---|---|
| `benchmark_serving.py` | `02ead008104e89a0f8d5a309d31ffbe8443cd7614012d1bacc90734346153bde` |
| `backend_request_func.py` | `c2ec9a4f566394835634db81ae444d2a0a930355d3fc51d7d74cc7dc8cfe50a2` |
| `benchmark_utils.py` | `09ac4be616b2245f38f3106bc43fd5c641c8dd50a47786d7102ab288fe15de1a` |

许可证 Apache-2.0（见 `LICENSE`）。**为什么 vendored**：`vllm bench serve` 需要安装带编译扩展的 vllm 包
（数 GB CUDA 依赖），而这些脚本本身只依赖 aiohttp/transformers/tqdm/numpy/datasets，可以直接对任意
OpenAI 兼容服务跑。请求生成、客户端计时、指标聚合仍由 vLLM 代码完成；下面的补丁只适配 Strata 的 API 行为。

## 补丁清单（`patches/`，可 `patch -p1` 重放）

| # | 文件 | 改了什么 | 为什么必须 |
|---|---|---|---|
| 1 | `backend_request_func.py` | 跳过非 `data:` 的 SSE 行 | Strata 会发 `: keep-alive` 注释；原版直接 `json.loads` 会崩 |
| 2 | `backend_request_func.py` | TTFT/ITL 只在**有 content** 的 chunk 上计时；usage chunk 单独收 | 第一个 chunk 是 role 空壳、最后一个可能只带 usage；原版会把 TTFT 记成 0 附近、ITL 夹进空 chunk |
| 3 | `backend_request_func.py` | chat payload 增加 `"chat_template_kwargs":{"enable_thinking":false}` | 不关思考时，TTFT 测到的是「思考结束」而不是 prefill；decode 速率不受影响 |
| 4 | `benchmark_serving.py` | 随机数据从「随机 token id → decode」改成「固定词表生成文本 → encode 截断」 | 原版对 byte-level BPE 的 decode→encode 不可逆：nominal 1024 token 的 prompt 被 Strata 重新数成 **3,346** token，prefill 长度全错 |
| 5 | `benchmark_serving.py` | warmup 用**不同**的短 prompt（1K 档用 512 token，其余 ≤2048；≤8 输出） | 原版 warmup 复用 `input_requests[0]`，主测再发同一 prompt → Strata conversation cache 99% 命中（实测 prompt 1036、reused 1029） |

补丁 4/5 会改变「随机」的含义：prompt 是固定 200 词词表拼出的可读性很差的文本，但 token 数与预填长度
精确、每次 seed 不同、不命中前缀缓存。这比原版更接近「测引擎」而不是「测 tokenizer 往返」。

## 重放 / 升级

```bash
# 从上游取原文件后：
patch -p1 -d <dir> < patches/0001-backend_request_func.patch
patch -p1 -d <dir> < patches/0001-benchmark_serving.patch
```

换 vLLM 版本时**不要直接套**：先确认 `backend_request_func.py` 的 SSE 解析与 payload 结构没变，
再按上表逐条重新落。结果对比必须注明「vLLM 0.7.3 + 本目录 5 个补丁」，不能写成 stock vLLM。
