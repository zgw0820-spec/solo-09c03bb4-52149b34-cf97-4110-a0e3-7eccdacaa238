# 风机停机预警训练平台（可续跑）

用 FastAPI + NumPy + SQLite 实现的风机停机预警逻辑回归训练平台。训练由两个工作器
容器执行，每个批次都把权重、动量、损失和进度作为**一个原子断点**提交；工作器退出后
另一个工作器自动接管，并从最近断点继续。最终权重与不中断训练逐位一致。

## 组成

| 组件 | 说明 |
| --- | --- |
| `api` | FastAPI，提交任务、查询进度/断点/权重、Swagger 演示 |
| `worker-1` / `worker-2` | 同一镜像的 `python -m app.worker`，共享同一个 SQLite 文件 |
| `training-data` | Docker 卷，挂载到三个容器的 `/data` |

两个工作器共享**同一个** SQLite 数据库（`/data/training.db`），因此"检查租约并写入
断点"和"写入断点"是同一个 `BEGIN IMMEDIATE` 事务，不需要跨库协调。

## 训练约定

- 输入 8～64 条记录，每条是风机振动/电流等特征，标签 `0`=正常、`1`=停机预警；
  各特征列按提交样本标准化后追加偏置列，权重长度为 `特征数 + 1`。
- 轮数 2～5；按输入顺序每 8 条一批（最后一批可不足 8 条）。
- 逻辑回归（sigmoid + 二元交叉熵），SGD，学习率 `0.1`，动量 `0.9`，float64。
- 每批提交的断点包含：`step`、`training_epoch`、`batch_in_epoch`、`weights`、
  `momentum_velocity`、`batch_loss`、`loss`（全量交叉熵）、`progress`、提交时间、
  `worker_id`、`lease_epoch`。
- 每个任务都保存各特征列的**原始均值、标准差以及常量列标记**（`column_stats`），
  用于跨样本换算权重。

## 续训（派生任务）

同型号设备补入新标注样本后，可从**已完成**任务派生一个续训任务：

```
POST /jobs/{source_job_id}/continuations
{"features": [[...], ...], "labels": [...], "epochs": 3}
```

- 新样本必须是 8～64 条、**特征列数与源任务相同**；按新样本自身的统计量标准化。
- 源权重换算到新样本的标准化尺度：

  `c_i = a_i * st_i / ss_i`，`d = b + Σ a_i * (mt_i - ms_i) / ss_i`

  因此派生任务在第一次更新前，对新样本**原始测值**的 logit 与源模型完全一致；
  新样本中的常量列（标准化后恒为 0）其整段源贡献并入偏置，该列权重保持 0。
  常量源列权重必然为 0，无需换算。
- 动量从零开始（`momentum_velocity = [0.0, ...]`）。
- 新任务保留 `source_job_id`、`initial_weights`、`initial_weights_source`
  （`scaled_from_source` / `zeros`）以及 `job_derived` 审计事件。
- 源任务缺少原始统计量（`column_mean_json` 等为 NULL，例如旧库）时拒绝派生：
  从已标准化的设计矩阵反推统计量只是猜测，不采用。
- 源任务不存在返回 404；未完成、维度不符、样本非法返回 422（含源任务缺少统计量）。
- 源任务只读，派生与续训都不会修改源任务；派生任务照常走租约、批次断点，
  工作器退出后从已确认断点续训，最终权重与不中断时逐位一致。

派生任务同样是普通任务，`GET /jobs/{id}`、`/checkpoints`、`/events` 行为不变。
同一源可派生多次，也可链式派生（用派生任务再派生）。

## 启动

```bash
docker compose up -d --build
# Swagger: http://localhost:8000/docs
```

`worker-1` / `worker-2` 故意不设 `restart` 策略，便于演示接管。被 kill 的工作器用
`docker compose start worker-1` 恢复（它会以空闲状态重新加入，不会抢占已完成任务）。

## API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/jobs` | 提交 `{"features": [[...]], "labels": [...], "epochs": 3}`，返回 201 与任务 |
| `POST` | `/jobs/{source_job_id}/continuations` | 从已完成任务派生续训任务，body 同 `/jobs`，返回 201 与新任务 |
| `GET` | `/jobs` | 最近任务列表 |
| `GET` | `/jobs/{job_id}` | 已确认步数、进度、损失、当前权重；完成时含 `final_weights`；含 `column_stats`、`source_job_id`、`initial_weights` |
| `GET` | `/jobs/{job_id}/checkpoints` | 全部批次断点（可按 `after_step` 增量拉取） |
| `GET` | `/jobs/{job_id}/events` | 审计事件：创建、派生、开始/从第几步续训、完成、丢失租约 |
| `GET` | `/workers` | 工作器心跳与租约持有者 |
| `GET` | `/health` | 健康检查 |

`Swagger`（`/docs`）里直接 `POST /jobs` 提交即可看到训练开始；等任务完成后用
`POST /jobs/{id}/continuations` 提交新样本即可看到续训。`GET /jobs/{id}` 与
`GET /jobs/{id}/checkpoints` 查看进度与每个断点的权重/动量/损失。

## 续跑与防回退机制

1. 工作器每批在**同一个事务**里写检查点、更新任务进度、刷新租约心跳。任何一项失败
   则整体回滚，不会出现"权重已变但进度没变"的中间态。
2. 租约保存在同一数据库的 `leases` 表。工作器每 `1.5s` 心跳；持有者心跳超过
   `6s`（`TRAINING_LEASE_SECONDS`）未更新，另一个工作器才可接管，接管时
   `epoch + 1`。
3. 每次提交前在事务内校验 `(worker_id, epoch)` 仍是租约持有者。被接管后仍存活（或
   迟到唤醒）的旧工作器校验失败，直接 `LeaseLost` 退出：既不能回退断点，也不能把
   别人的完成状态改写，更不能重复提交完成事件。
4. 完成状态由 `step == total_steps` 在提交事务内一次性置位；旧工作器无法再写入。
5. SQLite 使用 WAL + `synchronous=FULL`，已提交的断点在进程崩溃/容器重启后仍然存在。

## 演示与验证

```bash
# 完整演示（Docker）：提交 -> 观察 -> kill 当前持有者 -> 接管续跑 -> 对比不中断训练
#              -> 派生续训 -> 再 kill -> 接管 -> 对比不中断的派生训练
python3 scripts/demo.py

# 没有 Docker 时用本机进程演示同样流程
python3 scripts/demo.py --local

# 不依赖 Docker 的四项自检
python3 scripts/selftest.py

# 单元/集成测试（含真实 uvicorn + 两个工作器进程、SIGKILL 接管、迟到工作器围栏、
# 派生续训的崩溃接管与逐位一致性）
python3 -m pytest -q
```

演示脚本第 6 步会用同一份输入再训练一次（不中断），并断言两次的 `final_weights`、
`loss`、`momentum_velocity` 完全相等；第 7～9 步派生一个续训任务、中途 kill 工作器，
再与不中断的派生训练对比 `initial_weights`、`final_weights` 与 `loss`。

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `TRAINING_DATA_DIR` | `./data` | 数据库目录 |
| `TRAINING_DB_PATH` | `$TRAINING_DATA_DIR/training.db` | 共享数据库 |
| `TRAINING_LEASE_SECONDS` | `6` | 租约超时 |
| `TRAINING_HEARTBEAT_INTERVAL` | `1.0` | 心跳间隔（必须小于租约） |
| `TRAINING_POLL_INTERVAL` | `0.3` | 空闲轮询/批间间隔 |
| `TRAINING_BATCH_DELAY_SECONDS` | `0` | 模拟每批耗时，便于演示 |
| `TRAINING_DEMO_PAUSE_FILE` | 未设置 | 演示用：文件存在时在批间暂停（不心跳） |
| `TRAINING_DEMO_CRASH_AFTER` | 未设置 | 演示用：指定步数提交后 `os._exit` |

## 目录

```
app/main.py       API 与 Swagger
app/worker.py     工作器主循环、租约续训
app/store.py      任务状态机、批次提交、断点查询、派生续训
app/lease.py      租约获取/续约/围栏校验
app/model.py      逻辑回归、交叉熵、SGD + 动量
app/features.py   输入校验、标准化、原始统计量与跨样本权重换算
app/db.py         SQLite 连接、表结构与增量迁移
tests/            单元、API、进程级集成测试
scripts/demo.py   Swagger 同款接口的端到端演示
scripts/selftest.py  无 Docker 自检
```
