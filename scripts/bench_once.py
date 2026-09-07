#!/usr/bin/env python3
"""单次客户端口径测量：prefill(8k 无缓存唯一文本) 或 decode(512)。
用法: python3 bench_once.py prefill|decode [次数]
输出: jsonl，每行 {type, prompt_tokens, completion_tokens, total_s, tps}
"""
import json, random, sys, time, urllib.request
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/home/gfyd-ai/LLM_MODEL/RadixArk--Qwen3.8-Flash-Next-NVFP4", trust_remote_code=True)
SEGS = ["量子计算纠错码与表面码阈值分析综述","多模态大模型视觉-语言对齐机制研究","分布式训练梯度压缩的收敛性保障",
        "长上下文检索增强生成的稀疏化策略","混合专家模型的负载均衡与路由学习","神经网络量化误差传播的统计分析",
        "扩散模型采样加速的确定性方法","图神经网络在分子性质预测中的应用","强化学习奖励塑形的安全性讨论",
        "联邦学习非独立同分布数据的鲁棒聚合","序列模型状态空间表示的线性代数基础","端侧推理引擎的算子融合与内存规划",
        "投机解码的接受率建模与优化","知识蒸馏的温度策略与类别不平衡","向量检索的乘积量化与倒排结构"]

def make_unique_prompt(n_tokens=8000):
    """每次调用生成不同随机文本（保证前缀缓存 miss）"""
    text, rng = "", random.Random()
    while len(tok.encode(text)) < n_tokens:
        text += rng.choice(SEGS) + str(rng.randint(0, 99999)) + "。"
    return text

def send(payload):
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.load(r)
    return body, time.time() - t0

def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "prefill"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    for i in range(n):
        if kind == "prefill":
            prompt = make_unique_prompt(8000)
            payload = {"model": "Qwen3.8-27B", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 32, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
        else:
            payload = {"model": "Qwen3.8-27B", "messages": [{"role": "user", "content": "写一篇关于人工智能发展的长文"}],
                       "max_tokens": 512, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
        body, dt = send(payload)
        if "error" in body:
            print(json.dumps({"type": kind, "error": str(body["error"])[:100]}), flush=True)
            continue
        u = body["usage"]
        row = {"type": kind, "prompt": u["prompt_tokens"], "completion": u["completion_tokens"],
               "total_s": round(dt, 2), "tps": round(u["prompt_tokens"] / dt, 1) if kind == "prefill" else round(u["completion_tokens"] / dt, 1)}
        print(json.dumps(row), flush=True)
        time.sleep(3)  # 间隔避免 radix/统计干扰

if __name__ == "__main__":
    main()
