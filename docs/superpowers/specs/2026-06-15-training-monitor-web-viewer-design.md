# 训练进程 Web 看板(只读) — 设计 Spec

> 日期:2026-06-15 · 仓库 U-Bench · 分支:见落地时新建
> 状态:设计已确认,待写实现 plan。

## 1. 目标与范围

给 U-Bench 加一个**只读**的 Web 看板,实时显示**当前正在跑的训练 job**:它在哪张卡、跑到第几 epoch、当前/最佳 val_IoU,以及每张 GPU 的利用率/显存/温度。用于在两卡上做配方阶梯扫描时"抬头就能看到全局状态"。

**明确不做(YAGNI):**
- 不从页面控制 job(不 kill、不启动)——纯只读。
- 不加任何第三方依赖(纯 Python 标准库 + 原生 JS)。
- 不改动任何训练代码(`main.py`、dataloader、模型、adapter 一律不碰)。
- 不做用户认证、不做历史指标数据库、不做曲线图(进度用进度条 + 数字即可)。

## 2. 数据源(已核实)

每个 run 落在 `output/<model>/<dataset>/<exp_name>/`:
- `config.json`:权威超参,含 `model / dataset_name / gpu / max_epochs / seed / base_lr / batch_size / exp_name / exp_save_dir`。
- `training.log`:逐 epoch 追加,格式已核实:
  - 进行中:`<ts> - INFO - epoch [94/100]  train_loss: 3.37, train_iou: 0.73 - val_loss 0.38 - val_iou 0.6163 - val_SE .. - val_PC .. - val_F1 .. - val_ACC ..`
  - 收尾:`<ts> - INFO - Training completed. Best IoU: 0.6228, Best Epoch: 83, ...`
- 进程命令行(`ps`)含 `--gpu N`,直接给出物理卡号(因 `main.py` 在 torch 初始化前设 `CUDA_VISIBLE_DEVICES=--gpu`,再硬绑 `cuda:0`)。

GPU 实时状态来自 `nvidia-smi` 查询。

## 3. 架构

新增目录 `tools/monitor/`,逻辑与 web 分离(逻辑可单测):

```
tools/monitor/
  __init__.py
  __main__.py        # 使 `python -m tools.monitor` 可启动
  collector.py       # 纯逻辑:采集 + 解析 + join,无 HTTP
  server.py          # stdlib ThreadingHTTPServer,两个路由
  index.html         # 单页 vanilla JS 看板
```

### 3.1 `collector.py`(可测核心)

对外主函数 `list_jobs(output_root: str) -> dict`,返回:
```python
{
  "generated_at": <iso8601>,
  "gpus": [ {"index":0,"name":..,"util":0-100,"mem_used":MB,"mem_total":MB,"temp":C}, ... ],
  "jobs": [ {
      "pid":int, "gpu":int|None, "exp_name":str, "model":str, "dataset":str,
      "uptime_sec":int, "cpu":float, "mem":float,
      "epoch":int|None, "total_epochs":int|None, "val_iou":float|None,
      "best_iou":float|None, "best_epoch":int|None,
      "status":"running"|"starting"|"finished"|"stale"|"no_log",
      "seed":int|None, "base_lr":float|None, "log_mtime":<iso8601>|None
  }, ... ]
}
```

四步:
1. **进程发现**:跑 `ps` 列出 `python … main.py` 进程,取 PID、etime→`uptime_sec`、%cpu、%mem、完整 cmdline;从 cmdline 解析 `--model/--dataset_name/--exp_name/--gpu/--max_epochs/--seed/--base_lr`(缺省回退到 config.json)。
2. **进度解析**:由 `(model,dataset,exp_name)` 拼 `output_root/<model>/<dataset>/<exp_name>/`,读 `config.json`(权威超参)+ tail `training.log`(读末尾 N KB 即可)正则提取最后一条 `epoch [X/Y]` 与 `val_iou`、最近的 `Best IoU/Best Epoch`;若含 `Training completed` → `finished`。
3. **GPU 状态**:`nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits` 解析为 `gpus`;另查 `--query-compute-apps=pid,used_memory` 做 PID↔卡交叉校验(主映射仍用 cmdline 的 `--gpu`)。
4. **join + 状态判定**:
   - `gpu` 取 cmdline `--gpu`(单值);多值场景本仓库不出现,取第一个。
   - `status`:有 `Training completed` → `finished`;进程在但无 `training.log` → `no_log`;有 log 但无 epoch 行 → `starting`;`log_mtime` 距今 > `STALE_SEC`(默认 180s)→ `stale`;否则 `running`。

**命令执行注入**:`list_jobs` 内部经一个 `run_cmd(args)->str` 间接调用 `ps/nvidia-smi`,测试时可注入桩,无需真实进程/显卡。

### 3.2 `server.py`

- `ThreadingHTTPServer`,`BaseHTTPRequestHandler`:
  - `GET /` → 返回 `index.html`(读同目录文件)。
  - `GET /api/jobs` → `200 application/json`,body=`json.dumps(collector.list_jobs(output_root))`。
  - 其它 → 404。
- CLI(argparse):`--host`(默认 `127.0.0.1`)、`--port`(默认 `8800`)、`--output-root`(默认 `./output`)。
- 每个请求实时采集(无后台线程/缓存);采集耗时主要是两次 subprocess,可接受。

### 3.3 `index.html`

- 纯原生 JS,无构建、无 CDN 依赖。
- 顶部:每张 GPU 一个卡片——名称、util% 条、显存 used/total 条、温度。
- 下方:job 列表,每个 job 一行/卡:`exp_name` · `model·dataset` · `GPU#` · `PID` · 运行时长 · **epoch 进度条 X/Y** · 当前 val_IoU · best IoU@ep · seed/lr;`status` 配色(running 绿 / starting 黄 / finished 灰 / stale·no_log 红)。
- `setInterval(fetchJobs, 4000)` 轮询 `/api/jobs`;请求失败显示"连接丢失"横幅但不清空上次数据。

## 4. 数据流

浏览器 → 每 4s `GET /api/jobs` → server 调 `collector.list_jobs(output_root)` → `run_cmd` 跑 `ps`+`nvidia-smi`、读 `config.json`+`training.log` → 组装 JSON → JS 渲染。

## 5. 健壮性

- `nvidia-smi` 不存在/报错 → `gpus: []`,jobs 仍正常列(`gpu` 来自 cmdline)。
- 进程在跑但 exp 目录/日志缺失 → `status:"no_log"`,不报错。
- `config.json` 缺字段 → 用 cmdline 回退;都没有则该字段 `None`。
- 所有 subprocess 调用包 `try/except` + 超时(如 5s),任何子步骤失败都降级,**`/api/jobs` 端点永不 500**。
- `training.log` 只读末尾片段,避免大文件全量读。

## 6. 测试(pytest,`tests/`)

- `collector` 注入 `run_cmd` 桩:
  - 喂第 2 节真实 `training.log` 样本文本 → 断言 `epoch=99,total=100,val_iou≈0.6194,best_iou≈0.6228,best_epoch=83`。
  - 喂含 `Training completed` 的样本 → `status=finished`。
  - 喂假 `ps` 输出(一条 `python main.py --model USEANet --dataset_name busi --exp_name X --gpu 1 ...`)→ 断言解析出 pid/gpu/exp_name 等。
  - 喂假 `nvidia-smi` CSV → 断言 `gpus` 解析。
  - `nvidia-smi` 抛异常 → 断言 `gpus==[]` 且不抛。
  - 状态判定:log_mtime 旧 → `stale`;无 epoch 行 → `starting`。
- 用 `tmp_path` 造 `output/<model>/<dataset>/<exp_name>/{config.json,training.log}` 验证目录定位与 join。

## 7. 足迹与启动

- **新增文件**:`tools/monitor/{__init__,__main__,collector,server}.py`、`tools/monitor/index.html`、`tests/test_monitor_collector.py`。
- **改动文件**:无(训练代码零改动)。
- **依赖**:无新增(纯标准库)。
- **启动**:`python -m tools.monitor --port 8800`,浏览器开 `http://127.0.0.1:8800`。

## 8. 可调参数(默认值,实现时可改)

- 端口 `8800`、host `127.0.0.1`、output-root `./output`、轮询 `4s`、`STALE_SEC=180`、tail 读取 `64KB`。
- 默认**只显示有进程在跑的 job**;`finished` 仅当其进程仍在(收尾瞬间)出现。是否额外列出"近期已完成的历史 run"——本期不做,留作后续。
