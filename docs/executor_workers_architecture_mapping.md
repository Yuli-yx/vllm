# vLLM V1：Executor-Workers 架构与源码对应说明

> 目的：将你提供的《图解 Vllm V1系列2：Executor-Workers架构》中的核心论点，与当前仓库源码做逐点映射，方便核对“文档描述 ↔ 实际实现”。
>
> 说明：你引用的文章链接基于 `v0.8.2` 代码路径；本文对照的是当前仓库版本（`/workspace/vllm`）。总体架构一致，但有少量默认参数差异（例如 MQ chunk 大小）。

---

## 1. 进程拆分：Client 与 EngineCore

文章观点：
- Client 负责 pre-process / post-process。
- EngineCore 负责调度和推理。
- 两者跨进程通信（ZMQ）。

源码对应：

### 1.1 LLMEngine 侧输入输出处理器

```python
# vllm/v1/engine/llm_engine.py
# Convert EngineInput --> EngineCoreRequest.
self.input_processor = InputProcessor(self.vllm_config, renderer)

# Converts EngineCoreOutputs --> RequestOutput.
self.output_processor = OutputProcessor(...)

# EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
self.engine_core = EngineCoreClient.make_client(...)
```

解读：
- `input_processor` / `output_processor` 明确对应文章说的前后处理模块。
- `EngineCoreClient.make_client(...)` 负责与 EngineCore 进程/实例通信。

### 1.2 EngineCoreProc 是 ZMQ 包装层

```python
# vllm/v1/engine/core.py
class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""
```

```python
# vllm/v1/engine/core.py  (输入线程片段)
with ExitStack() as stack, zmq.Context() as ctx:
    input_sockets = [
        make_zmq_socket(ctx, input_address, zmq.DEALER, identity=identity, bind=False)
        ...
    ]
    poller = zmq.Poller()
    ...
    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
```

解读：
- 文章关于 ZMQ 通信的描述与实现一致。

---

## 2. Executor 类型与选择逻辑

文章观点：
- `mp` / `ray` / `uni` / `external_launcher` 四类。
- 默认根据环境自动选择 backend。

源码对应：

### 2.1 backend 自动决策

```python
# vllm/config/parallel.py
if self.distributed_executor_backend is None and self.world_size_across_dp > 1:
    backend: DistributedExecutorBackend = "mp"
    ...
    elif self.data_parallel_backend == "ray":
        backend = "ray"
    elif ray_found:
        ...
    self.distributed_executor_backend = backend

if self.distributed_executor_backend is None and self.world_size == 1:
    self.distributed_executor_backend = "uni"
```

解读：
- 与文章“默认自动决策”一致。

### 2.2 backend 到类的映射

```python
# vllm/v1/executor/abstract.py
elif distributed_executor_backend == "ray":
    executor_class = RayDistributedExecutor  # 或 RayExecutorV2
elif distributed_executor_backend == "mp":
    executor_class = MultiprocExecutor
elif distributed_executor_backend == "uni":
    executor_class = UniProcExecutor
elif distributed_executor_backend == "external_launcher":
    executor_class = ExecutorWithExternalLauncher
```

解读：
- 与文章列举的 4 类完全一致。

---

## 3. Scheduler → Executor → Workers 控制流

文章观点：
- Scheduler 在 EngineCore 进程里。
- 每轮调度结果交给 Executor 广播给 Workers。

源码对应：

```python
# vllm/v1/engine/core.py
if not self.scheduler.has_requests():
    return {}, False
scheduler_output = self.scheduler.schedule()
future = self.model_executor.execute_model(scheduler_output, non_block=True)
...
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
```

解读：
- 调度输出 `scheduler_output` 直接传给 executor 执行。

---

## 4. MultiprocExecutor 初始化与队列结构

文章观点：
- `rpc_broadcast_mq`：Executor → Workers 输入广播。
- `worker_response_mq`：Worker → Executor 输出回传。
- 传递 handle 给子进程连接队列。

源码对应：

### 4.1 Executor 创建广播 MQ 并派生 workers

```python
# vllm/v1/executor/multiproc_executor.py
self.rpc_broadcast_mq = MessageQueue(
    self.world_size,
    self.local_world_size,
    max_chunk_bytes=max_chunk_bytes,
    connect_ip=mq_connect_ip,
)
scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

unready_worker_handle = WorkerProc.make_worker_process(
    ...,
    input_shm_handle=scheduler_output_handle,
    ...,
)
```

解读：
- 文章所说“mq handler 传给 worker”就是 `export_handle()` + `input_shm_handle`。

### 4.2 Worker 端连接输入、创建输出

```python
# vllm/v1/executor/multiproc_executor.py
# 接收 SchedulerOutput
self.rpc_broadcast_mq = MessageQueue.create_from_handle(input_shm_handle, self.worker.rank)

# 发送模型输出
self.worker_response_mq = MessageQueue(1, 1)
```

解读：
- 与文章“输入队列是连接、输出队列是 worker 自建”一致。

### 4.3 READY 握手 + busy loop 启动

```python
# vllm/v1/executor/multiproc_executor.py
ready_writer.send({
    "status": WorkerProc.READY_STR,
    "handle": worker.worker_response_mq.export_handle(),
    "peer_response_handles": worker.peer_response_handles,
})
...
worker.worker_busy_loop()
```

解读：
- 文章里“ready_socket + busy loop”在这里体现。

---

## 5. WorkerWrapper / Worker / ModelRunner 分层

文章观点：
- WorkerWrapper 负责 worker 生命周期与扩展。
- ModelRunner 持有模型、KV cache、attn backend 等，并执行推理。

源码对应：

### 5.1 WorkerWrapperBase 职责

```python
# vllm/v1/worker/worker_base.py
class WorkerWrapperBase:
    """
    ... responsible for lazily initializing the worker and handling the worker's lifecycle.
    """
```

```python
# vllm/v1/executor/multiproc_executor.py
wrapper = WorkerWrapperBase(...)
wrapper.init_worker(all_kwargs)
self.worker = wrapper
self.worker.init_device()
self.worker.load_model()
```

### 5.2 ModelRunner 负责 KV/Attention 与执行

```python
# vllm/v1/worker/gpu/model_runner.py
self.block_tables = BlockTables(...)
self.attn_backends, self.attn_groups, attn_cg_support = init_attn_backend(...)
```

```python
# vllm/v1/worker/gpu/model_runner.py
def execute_model(...):
    input_batch = self.prepare_inputs(...)
    block_tables, slot_mappings = self.prepare_attn(input_batch)
    attn_metadata = self.model_state.prepare_attn(...)
    ...
```

解读：
- 与文章描述一致，且代码颗粒度更细。

---

## 6. “(method, data) 广播执行”在代码里的具体形态

文章观点：
- Executor 广播 `(method, data)` 给 workers。
- Worker 执行并回传。

源码对应：

```python
# vllm/v1/executor/multiproc_executor.py
self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))
```

```python
# vllm/v1/executor/multiproc_executor.py
method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(indefinite=True)
if isinstance(method, str):
    func = getattr(self.worker, method)
elif isinstance(method, bytes):
    func = partial(cloudpickle.loads(method), self.worker)
output = func(*args, **kwargs)
```

解读：
- 与文中语义完全一致，只是实现上拆成 `method + args + kwargs + output_rank`。

---

## 7. 小数据 / 大数据双通道：ShmRingBuffer + ZMQ

文章观点：
- 小数据走共享内存 ring buffer。
- 大数据走 zmq socket。
- `buf[0]` 标识是否 overflow。

源码对应：

### 7.1 MessageQueue 本地读者路径

```python
# vllm/distributed/device_communicators/shm_broadcast.py
# local readers:
# 1. create a shared memory ring buffer to communicate small data
# 2. create a publish-subscribe socket to communicate large data
self.buffer = ShmRingBuffer(...)
self.local_socket = context.socket(XPUB)
```

### 7.2 enqueue 时的 overflow 分流

```python
# vllm/distributed/device_communicators/shm_broadcast.py
if total_bytes + len(all_buffers[0]) >= self.buffer.max_chunk_bytes:
    with self.acquire_write(timeout) as buf:
        buf[0] = 1  # overflow
    self.local_socket.send_multipart(all_buffers, copy=False)
else:
    with self.acquire_write(timeout) as buf:
        buf[0] = 0  # not overflow
        ...  # 写入 shm
```

### 7.3 dequeue 时按标记走 shm/socket

```python
# vllm/distributed/device_communicators/shm_broadcast.py
with self.acquire_read(timeout, indefinite) as buf:
    overflow = buf[0] == 1
    if not overflow:
        ...  # 从 shm 解包
if overflow:
    obj = MessageQueue.recv(self.local_socket, timeout)
```

解读：
- 与文章 3.1/3.2 的行为描述一一对应。

---

## 8. 与文章版本的可见差异（建议重点注意）

### 8.1 MQ chunk 默认阈值

文章中描述：`10MB`。  
当前仓库：默认 `16MB`（可通过环境变量配置）。

```python
# vllm/envs.py
"VLLM_MQ_MAX_CHUNK_BYTES_MB": lambda: int(
    os.getenv("VLLM_MQ_MAX_CHUNK_BYTES_MB", "16")
)
```

```python
# vllm/v1/executor/multiproc_executor.py
max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
```

> 结论：架构机制相同，默认参数随版本演进。

---

## 9. 快速核对清单（文档描述 ↔ 代码状态）

- [x] Client 前后处理与 EngineCore 解耦。
- [x] EngineCoreProc 通过 ZMQ 收发请求。
- [x] 四类 executor（mp/ray/uni/external_launcher）。
- [x] mp 模式下 Executor 广播输入 + Worker 回传输出。
- [x] Worker busy loop 持续执行 RPC。
- [x] 小包 shm ring、大包 zmq fallback。
- [x] overflow 标志 `buf[0]` 的判定与分支。
- [x] WorkerWrapper / Worker / ModelRunner 分层职责。

---

## 10. 建议你下一步深挖的源码入口

1. `vllm/v1/executor/multiproc_executor.py`
   - 看完整控制面（RPC 广播、ready/shutdown/故障处理）。
2. `vllm/distributed/device_communicators/shm_broadcast.py`
   - 看 ring buffer 并发协议与性能细节。
3. `vllm/v1/worker/gpu/model_runner.py`
   - 看数据如何落到 block table / slot mapping / attention metadata。

