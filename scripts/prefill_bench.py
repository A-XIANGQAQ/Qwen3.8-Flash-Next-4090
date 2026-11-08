#!/usr/bin/env python3
"""Prefill throughput: prompt_tokens / TTFT on uncached prompts (unique header per run)."""
import json, os, sys, time, urllib.request, statistics, random, string
BASE=os.environ.get("STRATA_BASE","http://127.0.0.1:8000")
MODEL=os.environ.get("STRATA_MODEL","qwen3.8-flash-next-q4kxl-ablit-l1")
PARA=("在分布式的推理系统里，专家并行的调度需要同时考虑显存占用、跨卡带宽和请求的到达分布。"
      "每一次前向传播都要决定哪些专家驻留在显卡上，哪些留在主机内存里，以及哪些需要临时换入。"
      "这些决定会直接影响首 token 的延迟，也会影响长上下文场景下整体的吞吐表现。")

def chat(prompt, max_tokens=8):
    body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":prompt}],
                     "max_tokens":max_tokens,"temperature":0.0,"stream":True,
                     "stream_options":{"include_usage":True}}).encode()
    req=urllib.request.Request(BASE+"/v1/chat/completions",data=body,
                               headers={"Content-Type":"application/json"})
    t0=time.perf_counter(); ttft=None; usage=None
    with urllib.request.urlopen(req,timeout=1200) as resp:
        for raw in resp:
            line=raw.decode("utf-8","replace").strip()
            if not line.startswith("data:"): continue
            p=line[5:].strip()
            if p=="[DONE]": break
            try: obj=json.loads(p)
            except json.JSONDecodeError: continue
            if obj.get("usage"): usage=obj["usage"]
            for ch in obj.get("choices") or []:
                d=ch.get("delta") or {}
                if (d.get("content") or d.get("reasoning_content")) and ttft is None:
                    ttft=time.perf_counter()-t0
    return {"ttft":ttft,"prompt_tokens":(usage or {}).get("prompt_tokens"),
            "completion_tokens":(usage or {}).get("completion_tokens")}

def hdr():
    return "".join(random.choice(string.ascii_lowercase) for _ in range(24))+" "

def build(target):
    k=1
    while True:
        r=chat(hdr()+PARA*k, 1)
        n=r["prompt_tokens"]
        if n>=target or k>400: return k,n
        k=max(2,int(k*target/max(n,1)*1.02))

def measure(target, reps=3):
    k,_=build(target); runs=[]
    for i in range(reps):
        r=chat(hdr()+PARA*k, 8)
        if r["ttft"] and r["prompt_tokens"]:
            runs.append({"prompt_tokens":r["prompt_tokens"],
                         "ttft_s":round(r["ttft"],3),
                         "prefill_tps":round(r["prompt_tokens"]/r["ttft"],1)})
    med=statistics.median([x["prefill_tps"] for x in runs]) if runs else None
    print(json.dumps({"target":target,"reps":runs,"median_prefill_tps":med},ensure_ascii=False))
    return med

if __name__=="__main__":
    out={}
    targets=[int(x) for x in sys.argv[1:]] or [6200, 12650]
    for t in targets:
        out[t]=measure(t)
    print("SUMMARY", json.dumps(out))
