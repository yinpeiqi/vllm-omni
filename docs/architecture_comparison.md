# vLLM-Omni 架构对比：V0 vs V1

## 目录
- [架构概览](#架构概览)
- [架构流程图对比](#架构流程图对比)
- [V0 架构详解](#v0-架构详解)
- [V1 架构详解](#v1-架构详解)
- [Diffusion 支持](#diffusion-支持)
- [核心差异对比](#核心差异对比)
- [优劣分析](#优劣分析)

---

## 架构概览

vLLM-Omni 提供两种架构实现：
- **V0**: 基于多进程 + 队列通信的传统架构
- **V1**: 基于 vLLM V1 EngineCore 的两线程 + asyncio.Queue 架构

---

## 架构流程图对比

以 **Omni 模型（LLM stage-0 → Diffusion stage-1）** 为例，对比两种架构的进程/协程关系。

### V0：多进程模型

以 Qwen Omni（Thinker LLM + Talker LLM + Vocoder）为例。

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ 主进程                                                                            │
│  ┌──────────────────────┐   ┌──────────────────────────────────────────────────┐ │
│  │ generate()           │   │ _output_handler()                                │ │
│  └──────────────────────┘   └──────────────────────────────────────────────────┘ │
└──────────┬─────────────────────────┬─────────────────────────┬───────────────────┘
  mp.Queue (in_q/out_q)    mp.Queue (in_q/out_q)    mp.Queue (in_q/out_q)
           ▼▲                        ▼▲                        ▼▲
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ Worker 进程-0         │  │ Worker 进程-1         │   │ Worker 进程-2        │
│ (Thinker LLM)         │  │ (Talker LLM)          │  │ (Vocoder)            │
│  ┌────────────────┐   │  │  ┌────────────────┐   │  │  ┌────────────────┐  │
│  │_stage_worker   │   │  │  │_stage_worker   │   │  │  │_stage_worker   │  │
│  │_async()        │   │  │  │_async()        │   │  │  │_async()        │  │
│  └────────────────┘   │  │  └────────────────┘   │  │  └────────────────┘  │
│  ┌────────────────┐   │  │  ┌────────────────┐   │  │  ┌────────────────┐  │
│  │output_handler()│   │  │  │output_handler()│   │  │  │output_handler()│  │
│  └────────────────┘   │  │  └────────────────┘   │  │  └────────────────┘  │
└──────────┬────────────┘  └──────────┬────────────┘  └──────────┬───────────┘
       ZMQ ▼ ▲ ZMQ               ZMQ ▼ ▲ ZMQ               ZMQ ▼ ▲ ZMQ
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│ EngineCore 进程-0    │   │ EngineCore 进程-1    │  │ EngineCore 进程-2    │
│ (Thinker)            │  │ (Talker)             │  │ (Vocoder)            │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

### V1：两线程 + asyncio 模型

以 Qwen Omni（Thinker LLM + Talker LLM + Vocoder）为例。

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ 主进程                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │ 主线程                                                                        │   │
│  │  ┌──────────────────────┐   ┌────────────────────────────────────────────┐   │   │
│  │  │ generate()           │   │ output_handler()                           │   │   │
│  │  └──────────────────────┘   └────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│         asyncio.Queue (request_queue) ▼  ▲ asyncio.Queue (output_queue)             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │ Orchestrator 后台线程                                                         │   │
│  │  ┌──────────────────────┐  ┌───────────────────────┐                         │   │
│  │  │ _request_handler()   │  │ _output_handler_loop()│                         │   │
│  │  └──────────────────────┘  └───────────────────────┘                         │   │
│  │  ┌──────────────────────────────────────────────────────────────────────┐    │   │
│  │  │ output_handler()  (stage-0 / stage-1 / stage-2)                      │    │   │
│  │  └──────────────────────────────────────────────────────────────────────┘    │   │
│  │       ZMQ ▼ ▲ ZMQ            ZMQ ▼ ▲ ZMQ            ZMQ ▼ ▲ ZMQ              │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
  │ EngineCore 进程-0    │  │ EngineCore 进程-1     │  │ EngineCore 进程-2    │
  │ (Thinker)            │  │ (Talker)             │  │ (Vocoder)            │
  └──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

---



### 核心设计：多进程 + 队列通信

```
主进程 (AsyncOmni/Omni)
  ↓
OmniBase._initialize_stages()
  ↓ 为每个 stage 创建 OmniStage 实例
  ↓ 为每个 stage 创建 in_q / out_q (mp.Queue)
  ↓
启动 stage worker 进程 (_stage_worker / _stage_worker_async_entry)
  ↓ 每个 worker 进程独立运行
  ↓ 在 worker 进程内初始化 OmniLLM / OmniDiffusion / AsyncOmniLLM / AsyncOmniDiffusion
  ↓ worker 从 in_q 读取任务，处理后写入 out_q
  ↓
主进程 orchestrator 轮询所有 stage 的 out_q
  ↓ try_collect() 非阻塞读取
  ↓ 处理输出，转发到下一 stage
```

### 关键组件

#### 1. OmniBase (omni.py)
**职责：** 管理所有 stage 的生命周期

**核心方法：**
- `_initialize_stages()`: 加载 stage 配置，创建 OmniStage 实例
- `_start_stages_v0()`: 为每个 stage 创建 mp.Queue，启动 worker 进程
- `_wait_for_stages_ready()`: 等待所有 stage 就绪
- `start_profile()` / `stop_profile()`: 性能分析控制

**关键属性：**
- `stage_list`: 所有 OmniStage 实例
- `_stage_in_queues` / `_stage_out_queues`: 每个 stage 的输入/输出队列
- `connectors`: Stage 间数据传输连接器
- `default_sampling_params_list`: 每个 stage 的默认采样参数

#### 2. AsyncOmni (async_omni.py)
**职责：** 异步 orchestrator，协调多 stage 流水线

**核心方法：**
- `generate()`: async generator 接口，协调整个流水线
- `_run_output_handler()`: 启动后台任务轮询所有 stage 的 out_q
- `_process_sequential_results()`: 顺序处理 stage 输出（stage 间串行）
- `_process_async_results()`: 异步处理 stage 输出（stage 间并行）
- `_process_single_result()`: 处理单个 stage 的输出

**关键属性：**
- `request_states`: 跟踪每个 request 的状态 (ClientRequestState)
- `output_handler`: 后台任务，轮询所有 stage 的 out_q

#### 3. OmniStage (omni_stage.py)
**职责：** 封装单个 stage 的配置、状态和 worker 进程管理

**核心方法：**
- `attach_queues()`: 绑定 in_q / out_q
- `init_stage_worker()`: 启动 worker 进程
- `submit()`: 提交任务到 in_q
- `try_collect()`: 从 out_q 非阻塞读取结果
- `process_engine_inputs()`: 从上游 stage 的 outputs 构造本 stage 的输入
- `set_engine_outputs()`: 保存 engine outputs 供下游 stage 使用
- `stop_stage_worker()`: 停止 worker 进程

**关键属性：**
- `stage_id`: Stage 编号
- `stage_type`: "llm" 或 "diffusion"
- `engine_input_source`: 上游 stage ID 列表
- `final_output`: 是否产出最终结果
- `custom_process_input_func`: 自定义输入处理函数
- `_in_q` / `_out_q`: 输入/输出队列
- `_proc`: Worker 进程实例

#### 4. _stage_worker / _stage_worker_async (omni_stage.py)
**职责：** Worker 进程入口函数，运行 engine 并处理任务

**核心流程：**
1. 设备映射 (`set_stage_devices`)
2. 文件锁保证串行初始化 (避免 GPU OOM)
3. 初始化 engine (OmniLLM / OmniDiffusion / AsyncOmniLLM / AsyncOmniDiffusion)
4. 初始化 connectors (`build_stage_connectors`)
5. 发送 ready 信号
6. 批处理循环：
   - 从 in_q 读取任务
   - 通过 connector 接收数据 (`try_recv_via_connector`)
   - 调用 engine.generate()
   - 通过 shared memory 或直接序列化写入 out_q
   - 处理 profiler 控制命令

**关键特性：**
- 支持批处理 (max_batch_size)
- 支持 shared memory IPC (`maybe_dump_to_shm`)
- 支持 connector 接收数据
- 支持 profiler 控制

### 数据流

```
用户调用 AsyncOmni.generate()
  ↓
stage_list[0].submit(task) → in_q[0]
  ↓
worker-0 从 in_q[0] 读取
  ↓ 通过 connector 接收数据 (try_recv_via_connector)
  ↓ 调用 OmniLLM/AsyncOmniLLM.generate()
  ↓ 写入 out_q[0]
  ↓
_output_handler 轮询 out_q[0]
  ↓ _process_single_result()
  ↓ stage.set_engine_outputs()
  ↓ next_stage.process_engine_inputs() 构造下一 stage 输入
  ↓ try_send_via_connector() 通过 connector 发送
  ↓ stage_list[1].submit(task) → in_q[1]
  ↓
worker-1 从 in_q[1] 读取
  ↓ ... (重复)
  ↓
最终 stage 输出 → yield 给用户
```

---

## V1 架构详解

### 核心设计：两线程 + asyncio.Queue 通信

```
主线程 (AsyncOmniEngine)
  ↓ 薄代理
  ↓ stage-0 输入处理 (InputProcessor)
  ↓ request_queue (asyncio.Queue)
  ↓
后台线程 (Orchestrator)
  ↓ 独立 asyncio event loop
  ↓ 持有所有 StageAsyncCoreClient 实例
  ↓ _request_handler: 处理 add_request / abort / shutdown
  ↓ _output_handler_loop: 轮询所有 stage 的输出
  ↓ _forward_to_next_stage: stage 间转发
  ↓ output_queue (asyncio.Queue)
  ↓
主线程 try_get_output()
```

### 关键组件

#### 1. AsyncOmniEngine (async_omni_engine.py)
**职责：** 主线程薄代理，负责 stage-0 输入处理和对外接口

**核心方法：**
- `__init__()`: 启动 Orchestrator 后台线程，等待 ready 信号
- `add_request()`: **stage-0 输入处理在主线程完成**
  - 调用 `InputProcessor.process_inputs()` 生成 `EngineCoreRequest`
  - 若 stage-0 是 final_output，注册 `output_processors[0]`
  - 发送消息到 request_queue
- `try_get_output()`: 非阻塞读取 output_queue
- `try_get_output_blocking()`: 阻塞读取 output_queue
- `abort()`: 发送 abort 消息
- `shutdown()`: 发送 shutdown 消息

**关键属性：**
- `request_queue` / `output_queue`: asyncio.Queue，与 Orchestrator 通信
- `_orch_loop`: Orchestrator 的 event loop
- `input_processor`: InputProcessor 实例
- `output_processors`: 所有 stage 的 output processor 列表

**关键设计：**
- Stage-0 输入处理在主线程，避免 queue + coroutine-switch 开销
- 通过 `call_soon_threadsafe` 实现线程安全的 queue 操作

#### 2. Orchestrator (orchestrator.py)
**职责：** 后台线程，独立 event loop，管理所有 stage 和 stage 间转发

**核心方法：**
- `_initialize_stages()`: 串行初始化所有 stage
  - 调用 `prepare_engine_environment()` 全局环境设置
  - 初始化 connectors (`initialize_orchestrator_connectors`)
  - 对每个 stage:
    - 提取 metadata (`extract_stage_metadata`)
    - 设备映射 (`setup_stage_devices`)
    - 构建 vllm_config (`build_vllm_config`)
    - 文件锁保证串行初始化 (`acquire_device_locks` / `release_device_locks`)
    - 创建 `StageAsyncCoreClient`
    - 创建 tokenizer 和 output processor
- `run()`: 主入口，发送 ready 信号，启动 request_handler 和 output_handler
- `_request_handler()`: 处理来自主线程的消息 (add_request / abort / shutdown)
- `_output_handler_loop()`: 双路径输出处理
  - 轮询所有 stage: `_poll_stage_raw()` → `_process_stage_outputs()`
  - final_output stage: 发送给 client
  - 非 final stage: 用于 stage 间转发
- `_forward_to_next_stage()`: stage 间转发核心逻辑
  - 设置当前 stage 的 `engine_outputs`
  - 调用下一 stage 的 `process_engine_inputs()` 构建输入
  - 通过 `build_engine_core_request_from_tokens()` 构建轻量请求
  - 注册 output processor
  - 提交给下一 stage

**关键属性：**
- `stage_clients`: 所有 StageAsyncCoreClient 实例
- `output_processors`: 所有 stage 的 output processor (所有 stage 都有)
- `stage_vllm_configs`: 每个 stage 的配置
- `request_states`: 跟踪每个 request (OrchestratorRequestState)
- `connectors`: Stage 间数据传输连接器

**关键设计：**
- 所有 stage 都有 output processor，用于生成 RequestOutput
- 只有 final_output stage 的输出会发送给 client
- 非 final stage 的 RequestOutput 用于 stage 间转发

#### 3. StageAsyncCoreClient (stage_async_core_client.py)
**职责：** 单个 stage 的引擎客户端，继承 vLLM 的 AsyncMPClient

**核心方法：**
- `__init__()`: 调用 `super().__init__()` 复用 vLLM 的 EngineCore 架构
- `add_request_async()`: 添加请求
- `process_engine_inputs()`: 从上游 stage 的 `engine_outputs` 构造本 stage 的输入
  - 支持 `custom_process_input_func` 自定义处理
  - 默认：从 `engine_input_source` 获取上游 outputs，构建 `OmniTokensPrompt`
- `set_engine_outputs()`: 保存 engine outputs

**关键属性：**
- `stage_id` / `stage_type` / `engine_output_type`: Stage 元数据
- `final_output` / `final_output_type`: 是否产出最终结果
- `custom_process_input_func`: 自定义输入处理函数
- `engine_outputs`: 当前 stage 的输出

**关键设计：**
- 继承 `AsyncMPClient`，复用 ZMQ、EngineCore 进程管理
- 所有重初始化逻辑移至 `stage_init.py`

#### 4. stage_init.py
**职责：** Stage 初始化工具函数

**核心函数：**
- `extract_stage_metadata()`: 从 stage_config 提取轻量 StageMetadata
- `prepare_engine_environment()`: 全局环境设置（加载插件，设置 multiprocessing）
- `setup_stage_devices()`: 设备映射
- `build_vllm_config()`: 构建 vllm_config 和 executor_class
- `acquire_device_locks()` / `release_device_locks()`: 文件锁保证 stage 串行初始化 GPU

**StageMetadata 数据类：**
- `stage_id` / `stage_type` / `engine_output_type`
- `is_comprehension` / `requires_multimodal_data`
- `engine_input_source` / `final_output` / `final_output_type`
- `default_sampling_params` / `custom_process_input_func`
- `model_stage` / `runtime_cfg`

#### 5. build_engine_core_request_from_tokens (orchestrator.py)
**职责：** 轻量请求构建函数，用于 stage-1+ 的请求构建

**核心逻辑：**
- 跳过 tokenization，直接使用上游 token_ids
- 序列化 `prompt_embeds` (PromptEmbedsPayload)
- 序列化 `additional_information` (AdditionalInformationPayload)
- 设置 max_tokens (如果未设置)
- 返回 `OmniEngineCoreRequest`

**关键设计：**
- 避免重复 tokenization，优化性能
- 支持跨 stage 传递 embeddings 和额外信息

#### 6. InputProcessor (input_processor.py)
**职责：** Stage-0 的完整输入处理流水线

**核心方法：**
- `process_inputs()`: 处理输入 prompt
  - Tokenization
  - Multimodal 预处理
  - 序列化 prompt_embeds 和 additional_information
  - 生成 `EngineCoreRequest`

**关键设计：**
- 直接用 vLLM `InputProcessor`
- 仅在 stage-0 使用，stage-1+ 不执行

#### 7. MultimodalOutputProcessor (output_processor.py)
**职责：** 所有 stage 的输出处理

**核心方法：**
- `process_outputs()`: 处理 raw outputs，生成 RequestOutput
- `add_request()`: 注册 request

**关键设计：**
- 继承 vLLM `OutputProcessor`
- 所有 stage 都有 output processor
- 扩展 `OmniRequestState` 支持多模态 tensor 累积

### 数据流

```
用户调用 AsyncOmniV1.generate()
  ↓
AsyncOmniEngine.add_request()
  ↓ (主线程) InputProcessor.process_inputs()
  ↓ 生成 EngineCoreRequest
  ↓ 注册 output_processors[0]
  ↓ request_queue
  ↓
Orchestrator._handle_add_request()
  ↓ 记录 OrchestratorRequestState
  ↓ stage_clients[0].add_request_async()
  ↓
_output_handler_loop 轮询 stage_clients[0]
  ↓ _poll_stage_raw() → EngineCoreOutputs
  ↓ _process_stage_outputs() → RequestOutput
  ↓ 若非最终 stage:
    ↓ _forward_to_next_stage()
    ↓ set_engine_outputs()
    ↓ process_engine_inputs()
    ↓ build_engine_core_request_from_tokens()
    ↓ output_processors[1].add_request()
    ↓ stage_clients[1].add_request_async()
  ↓ 若 final_output stage:
    ↓ output_queue
    ↓
主线程 try_get_output()
  ↓ yield 给用户
```

---

## Diffusion 支持

### 背景

V0 原生支持 diffusion stage，V1 最初只支持 LLM stage（`extract_stage_metadata()` 对 `stage_type=="diffusion"` 直接抛 `NotImplementedError`）。现已在 V1 中完整实现 diffusion 支持。

### V0 Diffusion 实现

Diffusion stage 在 V0 中与 LLM stage 对称：独立 worker 进程，运行 `AsyncOmniDiffusion`，通过 `mp.Queue` 通信。

- stage-0 是 diffusion：原始 prompt 直接传入 worker，不经过 tokenization
- stage-N（LLM → Diffusion）：`custom_process_input_func` 将 LLM 输出转为 diffusion 输入
- pre/post processing 完全封装在 `DiffusionEngine.step()` 内部，主进程收到的已是 `OmniRequestOutput`

### V1 Diffusion 实现

V1 引入 `StageDiffusionClient`，在 Orchestrator 内对 diffusion stage 走独立路径，LLM 路径完全不变。

#### 改动文件

| 文件 | 改动 |
|------|------|
| `stage_init.py` | 删除 `NotImplementedError`，diffusion 提前 return `StageMetadata`（`engine_output_type=None`） |
| `stage_diffusion_client.py` | 新文件，封装 `AsyncOmniDiffusion`，暴露 Orchestrator 所需接口 |
| `orchestrator.py` | `_initialize_stages` diffusion 分支、`_output_handler_loop` 双路径 poll、`_build_stage_metrics` 覆盖 diffusion |
| `async_omni_engine.py` | stage-0 是 diffusion 时跳过 `OmniInputProcessor` |

#### StageDiffusionClient 接口

```python
class StageDiffusionClient:
    stage_type = "diffusion"

    async def add_request_async(request_id, prompt, sampling_params)
        # asyncio.create_task(_run(...))

    def get_diffusion_output_async() -> OmniRequestOutput | None
        # _output_queue.get_nowait()，空则返回 None

    async def abort_requests_async(request_ids)
    def shutdown()
```

#### 纯 Diffusion 数据流（单 stage-0）

```
generate(prompt, OmniDiffusionSamplingParams)
  → AsyncOmniEngine.add_request()        # 跳过 OmniInputProcessor
  → request_queue
  → Orchestrator._handle_add_request()
  → StageDiffusionClient.add_request_async()
  → asyncio.create_task(_run(...))
  → AsyncOmniDiffusion.generate()
      ThreadPoolExecutor → DiffusionEngine.step()
      → OmniRequestOutput{images}
  → _output_queue
  → _output_handler_loop: get_diffusion_output_async()
  → _build_stage_metrics() + _route_output()
  → output_queue
  → 主线程 yield OmniRequestOutput
```

#### LLM → Diffusion 数据流（两 stage）

```
stage-0 LLM 完成
  → _forward_to_next_stage()
  → custom_process_input_func(output, prompt, stage_clients)
  → diffusion_prompt (OmniPromptType)
  → StageDiffusionClient.add_request_async()
  → ... (同纯 diffusion 路径)
```

#### 关键设计决策

- **diffusion 无 OutputProcessor**：pre/post processing 封装在 `DiffusionEngine` 内，`AsyncOmniDiffusion.generate()` 直接返回 `OmniRequestOutput`，Orchestrator 无需额外处理
- **`finished` 永远为 True**：diffusion 一次性完成，`_route_output` 的 cleanup 逻辑天然兼容
- **metrics 正常上报**：`_build_stage_metrics` 在 diffusion 输出时同样调用，`count_tokens_from_outputs` 对 `OmniRequestOutput` 安全（token 数为 0）



| 维度 | V0 | V1 |
|------|----|----|
| **进程模型** | 多进程 (每个 stage 一个 worker 进程) | 两线程 (主线程 + Orchestrator 后台线程) |
| **通信方式** | `mp.Queue` (跨进程) | `asyncio.Queue` (跨线程) |
| **Orchestrator 位置** | 主进程中运行 | 后台线程中运行 (独立 event loop) |
| **Stage 实现** | `OmniStage` + worker 进程 (`_stage_worker`) | `StageAsyncCoreClient` (继承 `AsyncMPClient`) |
| **Engine 初始化** | 在 worker 进程内初始化 `OmniLLM` / `AsyncOmniLLM` | 在 `StageAsyncCoreClient` 内初始化 `EngineCore` |
| **Stage-0 输入处理** | 在 worker-0 进程内 | 在主线程 (`AsyncOmniEngine.add_request()`) |
| **Output Processor** | 在 worker 进程内 | 在 Orchestrator 线程内 (所有 stage 都有) |
| **Stage 间转发** | 主进程 orchestrator 轮询 out_q，调用 `process_engine_inputs()`，通过 connector 发送 | Orchestrator 线程内，调用 `process_engine_inputs()`，构建轻量请求 |
| **请求构建** | 完整 input processing (在 worker 内) | Stage-0: 完整 input processing (主线程)<br>Stage-1+: 轻量构建 (`build_engine_core_request_from_tokens`) |
| **Connector 使用** | 强制使用 connector 传输数据 | 当前实现中 disabled (TODO) |
| **并发模型** | 多进程并发 | 单进程多线程 + asyncio |
| **IPC 机制** | mp.Queue + shared memory | asyncio.Queue (线程间) + ZMQ (EngineCore 进程间) |
| **批处理** | 在 worker 进程内批处理 | 在 EngineCore 内批处理 |
| **设备锁** | 在 worker 进程内获取 | 在 Orchestrator 初始化时获取 |

---

## 优劣分析

### V0 优势
1. **进程隔离**：每个 stage 独立进程，稳定性更好，一个 stage 崩溃不影响其他 stage
2. **分布式支持**：支持 Ray backend，可以跨机器部署
3. **Connector 机制完整**：强制使用 connector 传输数据，支持共享内存等高效传输方式
4. **灵活性高**：可以独立控制每个 stage 的资源和配置
5. **成熟稳定**：已经在生产环境验证

### V0 劣势
1. **多进程开销大**：进程创建、销毁、上下文切换开销
2. **IPC 复杂**：mp.Queue + shared memory，序列化/反序列化开销
3. **Orchestrator 轮询开销**：主进程轮询所有 stage 的 out_q，CPU 占用高
4. **内存占用大**：每个 stage 独立进程，内存隔离导致总内存占用大
5. **调试困难**：多进程调试复杂

### V1 优势
1. **两线程模型**：开销小，上下文切换快
2. **Stage-0 输入处理优化**：在主线程完成，避免 queue + coroutine-switch 开销
3. **复用 vLLM V1 基础设施**：继承 `AsyncMPClient` 和 `EngineCore`，代码复用度高
4. **轻量请求构建**：stage-1+ 跳过 tokenization，优化性能
5. **统一 output processor**：所有 stage 都有 output processor，逻辑一致
6. **调试友好**：单进程多线程，调试简单

### V1 劣势
1. **线程模型隔离性差**：一个 stage 崩溃可能影响整个进程
2. **Connector 机制未完整实现**：当前 disabled，需要补充
3. **不支持 Ray backend**：无法跨机器部署
4. **GIL 限制**：Python GIL 可能成为瓶颈（但 EngineCore 在独立进程，影响有限）
5. **成熟度较低**：V1 架构较新，生产环境验证不足

---

## 关键目录结构

```
vllm_omni/
├── engine/                          # V1 引擎核心
│   ├── async_omni_engine.py         # AsyncOmniEngine (主线程薄代理)
│   ├── orchestrator.py              # Orchestrator (后台线程)
│   ├── stage_async_core_client.py   # StageAsyncCoreClient (LLM stage)
│   ├── stage_diffusion_client.py    # StageDiffusionClient (Diffusion stage)
│   ├── stage_init.py                # Stage 初始化工具
│   ├── input_processor.py           # InputProcessor (stage-0 LLM)
│   ├── output_processor.py          # MultimodalOutputProcessor
│   ├── arg_utils.py                 # OmniEngineArgs
│   └── __init__.py                  # 数据类定义
├── entrypoints/                     # 入口点
│   ├── async_omni_v1.py             # AsyncOmniV1 (V1 入口)
│   ├── async_omni.py                # AsyncOmni (V0 入口)
│   ├── async_omni_diffusion.py      # AsyncOmniDiffusion (V0 diffusion engine)
│   ├── omni.py                      # Omni / OmniBase (V0 基类)
│   ├── omni_stage.py                # OmniStage + _stage_worker
│   └── utils.py                     # 配置加载工具
├── distributed/                     # 跨 stage 连接器
│   └── omni_connectors/
├── inputs/                          # 输入数据类型
└── outputs.py                       # OmniRequestOutput
```

---

## 总结

V1 是针对 vLLM V1 架构的优化版本，主要优化点在于：
1. 减少进程间通信开销（两线程 vs 多进程）
2. Stage-0 输入处理在主线程（避免 queue 开销）
3. 轻量请求构建（跳过 tokenization）
4. 复用 vLLM V1 基础设施（AsyncMPClient、EngineCore）

V0 更成熟稳定，支持分布式部署，适合生产环境。V1 性能更优，但成熟度较低，适合新项目和单机部署。
