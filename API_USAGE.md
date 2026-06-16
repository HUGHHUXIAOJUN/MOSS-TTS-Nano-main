# Web API 调用说明

服务启动后默认监听：

```text
http://127.0.0.1:8000
```

容器内监听 `0.0.0.0:18083`，宿主机通过 compose 映射到 `8000`。

## 推荐调用方式

外部项目优先使用普通生成接口：

```text
POST /api/generate
Content-Type: multipart/form-data
```

这个接口会等生成完成后一次性返回 JSON，其中 `audio_base64` 是完整 WAV 文件的 base64 字符串。调用方只需要解码并保存为 `.wav`，最容易接入。

流式接口适合浏览器或实时播放场景：

```text
POST /api/generate-stream/start
GET  /api/generate-stream/{stream_id}/audio
GET  /api/generate-stream/{stream_id}/status
GET  /api/generate-stream/{stream_id}/result
POST /api/generate-stream/{stream_id}/close
```

流式音频是原始 `pcm_s16le` 字节流，不是 WAV。

如果外部项目是后端服务调用这个 TTS 服务，直接按 HTTP 调用即可。如果外部项目是浏览器前端，且页面地址和 TTS 服务不是同源，例如前端在 `http://localhost:3000`、TTS 在 `http://localhost:8000`，当前服务没有配置 CORS，浏览器会拦截跨域请求。此时建议：

- 让你的后端转发调用 TTS 服务。
- 或在 `app.py` 里增加 FastAPI `CORSMiddleware` 白名单。

## 服务状态

检查服务进程是否可访问：

```bash
curl http://127.0.0.1:8000/health
```

检查模型预热是否完成：

```bash
curl http://127.0.0.1:8000/api/warmup-status
```

`/api/warmup-status` 返回：

```json
{
  "state": "ready",
  "progress": 1.0,
  "message": "Warmup complete...",
  "error": null,
  "ready": true,
  "failed": false,
  "status_text": "Warmup complete..."
}
```

判断规则：

- `ready: true`：可以发起生成。
- `failed: true`：模型加载或预热失败，看 `error`。
- `state` 为 `pending` 或 `running`：还在加载/预热。

即使不主动检查，`/api/generate` 和 `/api/generate-stream/start` 也会等待 warmup；首次请求可能很慢。

## 普通生成接口

### 请求

```text
POST /api/generate
Content-Type: multipart/form-data
```

必填字段：

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `text` | string | 要合成的文本。字段必须存在。 |
| `prompt_audio` | file | 参考音频。和 `demo_id` 二选一。 |
| `demo_id` | string | 使用内置 demo 参考音频。和 `prompt_audio` 二选一。 |

推荐外部项目上传 `prompt_audio`，不要依赖内置 demo。内置 demo 的 id 是启动时按 `assets/demo.jsonl` 顺序生成的 `demo-1`、`demo-2` 等。

可选字段：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `max_new_frames` | `375` | 最大生成音频帧数，越大越可能生成更长音频。 |
| `voice_clone_max_text_tokens` | `75` | 长文本切块 token 上限。 |
| `tts_max_batch_size` | `0` | TTS 批大小，`0` 表示自动。CPU 上建议 `1`。 |
| `codec_max_batch_size` | `0` | Codec 批大小，`0` 表示自动。 |
| `enable_text_normalization` | `1` | 是否启用 WeTextProcessing，`1`/`0`。 |
| `enable_normalize_tts_text` | `1` | 是否启用内置文本清理，`1`/`0`。 |
| `cpu_threads` | `0` | CPU 线程数，`0` 表示使用服务默认值。 |
| `attn_implementation` | `model_default` | `model_default`、`sdpa`、`eager`。CPU Web 服务通常会映射到 `eager`。 |
| `do_sample` | `1` | 是否采样，`1`/`0`。 |
| `text_temperature` | `1.0` | 文本层采样温度。 |
| `text_top_p` | `1.0` | 文本层 top-p。 |
| `text_top_k` | `50` | 文本层 top-k。 |
| `audio_temperature` | `0.8` | 音频层采样温度。 |
| `audio_top_p` | `0.95` | 音频层 top-p。 |
| `audio_top_k` | `25` | 音频层 top-k。 |
| `audio_repetition_penalty` | `1.2` | 音频重复惩罚。 |
| `seed` | `0` | `0` 或空字符串表示随机；其他整数表示固定种子。 |

### curl 示例：上传参考音频

```bash
curl -X POST http://127.0.0.1:8000/api/generate \
  -F "text=你好，这是一次外部项目调用测试。" \
  -F "prompt_audio=@./prompt.wav" \
  -F "voice_clone_max_text_tokens=75" \
  -F "max_new_frames=375" \
  -F "enable_text_normalization=1" \
  -F "enable_normalize_tts_text=1"
```

### curl 示例：使用内置 demo 音频

```bash
curl -X POST http://127.0.0.1:8000/api/generate \
  -F "text=你好，这是一次外部项目调用测试。" \
  -F "demo_id=demo-1"
```

### 成功响应

HTTP `200`：

```json
{
  "audio_base64": "UklGRiQ...",
  "sample_rate": 48000,
  "run_status": "Done | mode=voice_clone | prompt=prompt | ...",
  "prompt_audio_path": "Uploaded: prompt.wav",
  "warmup_status_text": "Warmup complete...",
  "text_normalization_status_text": "WeTextProcessing ready. languages=zh,en",
  "text_chunks": ["你好，这是一次外部项目调用测试。"],
  "normalized_text": "你好，这是一次外部项目调用测试。",
  "normalization_method": "robust_pre+wetext+robust_post",
  "text_normalization_language": "zh"
}
```

判断成功：

- HTTP 状态码是 `200`。
- JSON 中没有 `error`。
- `audio_base64` 存在且非空。

`audio_base64` 是 WAV 文件内容，直接 base64 解码保存即可。

### 错误响应

常见错误：

```json
{"error": "demo_id is required unless prompt speech is uploaded."}
```

```json
{"error": "text is required."}
```

```json
{"error": "Warmup failed: ..."}
```

判断失败：

- HTTP 状态码不是 `2xx`。
- 或 JSON 中包含 `error`。

## Python 调用示例

```python
import base64
from pathlib import Path

import requests

base_url = "http://127.0.0.1:8000"

with Path("prompt.wav").open("rb") as audio_file:
    response = requests.post(
        f"{base_url}/api/generate",
        data={
            "text": "你好，这是一次 Python 项目调用测试。",
            "max_new_frames": "375",
            "voice_clone_max_text_tokens": "75",
            "enable_text_normalization": "1",
            "enable_normalize_tts_text": "1",
        },
        files={
            "prompt_audio": ("prompt.wav", audio_file, "audio/wav"),
        },
        timeout=600,
    )

response.raise_for_status()
payload = response.json()

if payload.get("error"):
    raise RuntimeError(payload["error"])
if not payload.get("audio_base64"):
    raise RuntimeError("missing audio_base64")

Path("output.wav").write_bytes(base64.b64decode(payload["audio_base64"]))
print(payload["run_status"])
```

## JavaScript 调用示例

```js
async function generateSpeech({ baseUrl, text, promptAudioFile }) {
  const form = new FormData();
  form.append("text", text);
  form.append("prompt_audio", promptAudioFile);
  form.append("max_new_frames", "375");
  form.append("voice_clone_max_text_tokens", "75");
  form.append("enable_text_normalization", "1");
  form.append("enable_normalize_tts_text", "1");

  const response = await fetch(`${baseUrl}/api/generate`, {
    method: "POST",
    body: form,
  });

  const payload = await response.json();
  if (!response.ok || payload.error) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  if (!payload.audio_base64) {
    throw new Error("missing audio_base64");
  }

  const binary = atob(payload.audio_base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: "audio/wav" });
}
```

## 流式生成接口

流式生成适合边生成边播放。调用顺序如下。

### 1. 创建流式任务

```text
POST /api/generate-stream/start
Content-Type: multipart/form-data
```

参数和 `/api/generate` 相同。

成功返回：

```json
{
  "stream_id": "stream-...",
  "audio_url": "/api/generate-stream/stream-.../audio",
  "status_url": "/api/generate-stream/stream-.../status",
  "result_url": "/api/generate-stream/stream-.../result",
  "sample_rate": 48000,
  "channels": 2,
  "run_status": "Streaming realtime audio... exec=cpu",
  "prompt_audio_path": "Uploaded: prompt.wav",
  "text_chunks": ["..."],
  "normalized_text": "...",
  "normalization_method": "...",
  "text_normalization_language": "zh"
}
```

判断 start 成功：

- HTTP 状态码是 `200`。
- JSON 中有 `stream_id`、`audio_url`、`status_url`、`result_url`。

### 2. 读取音频流

```text
GET /api/generate-stream/{stream_id}/audio
```

响应头：

```text
Content-Type: application/octet-stream
X-Audio-Codec: pcm_s16le
X-Audio-Sample-Rate: 48000
X-Audio-Channels: 2
X-Stream-Id: stream-...
```

响应体是连续 PCM：

- 编码：signed 16-bit little-endian
- 声道：看 `X-Audio-Channels`
- 采样率：看 `X-Audio-Sample-Rate`
- 多声道为交错排列

如果要保存成文件，需要自己加 WAV header，或最终调用 result 接口拿 `audio_base64`。

### 3. 查询任务状态

```text
GET /api/generate-stream/{stream_id}/status
```

返回示例：

```json
{
  "stream_id": "stream-...",
  "state": "running",
  "run_status": "Streaming | emitted=1.20s | lead=0.30s",
  "error": null,
  "prompt_audio_path": "Uploaded: prompt.wav",
  "sample_rate": 48000,
  "channels": 2,
  "emitted_audio_seconds": 1.2,
  "lead_seconds": 0.3,
  "current_chunk_index": 0,
  "playback_chunk_index": 0,
  "text_chunks": ["..."],
  "first_audio_latency_seconds": 2.1,
  "completed_at": null,
  "ready": false,
  "failed": false,
  "closed": false,
  "status_text": "Streaming | emitted=1.20s | lead=0.30s",
  "stream_metrics": "state=running | emitted=1.20s | lead=0.30s | first_audio=2.10s"
}
```

判断规则：

- `ready: true` 或 `state: "done"`：生成完成。
- `failed: true` 或 `state: "failed"`：生成失败，看 `error`。
- `closed: true`：任务已被关闭。

### 4. 获取最终结果

```text
GET /api/generate-stream/{stream_id}/result
```

可能返回：

- HTTP `202`：任务还没完成，响应体是当前状态快照。
- HTTP `200`：任务完成，返回最终结果。
- HTTP `500`：任务失败。

完成时返回：

```json
{
  "stream_id": "stream-...",
  "ready": true,
  "state": "done",
  "prompt_audio_path": "Uploaded: prompt.wav",
  "run_status": "Done | mode=voice_clone | ...",
  "stream_metrics": "state=done | emitted=4.32s | lead=0.00s | first_audio=2.10s",
  "warmup_status_text": "Warmup complete...",
  "text_chunks": ["..."],
  "audio_chunk_ranges": [[0.0, 1.2, 0], [1.2, 2.5, 1]],
  "audio_base64": "UklGRiQ..."
}
```

`audio_base64` 是完整 WAV 文件内容。即使已经实时播放了 PCM 流，也可以用它保存最终音频。

### 5. 关闭任务

```text
POST /api/generate-stream/{stream_id}/close
```

建议在客户端取消播放、页面关闭、或拿到最终结果后调用。它会关闭任务并清理临时文件。

## 流式调用伪代码

```js
const start = await postMultipart("/api/generate-stream/start", formData);
const audioResponse = await fetch(start.audio_url);

// 并行：读取 audioResponse.body，按 pcm_s16le 播放
// 并行：轮询 start.status_url 判断 ready/failed

let result;
while (true) {
  const response = await fetch(start.result_url);
  if (response.status === 202) {
    await sleep(200);
    continue;
  }
  result = await response.json();
  if (!response.ok || result.error) {
    throw new Error(result.error || `HTTP ${response.status}`);
  }
  break;
}

if (result.ready && result.audio_base64) {
  // 保存最终 WAV
}

await fetch(`/api/generate-stream/${encodeURIComponent(start.stream_id)}/close`, {
  method: "POST",
});
```

## 音色库管理接口

页面里的“Voice Library / 音色库管理”使用下面这些接口。系统内置音色来自 `assets/demo.jsonl`，只能查询和生成，不能修改或删除；自定义音色会保存录音文件和元数据，之后会出现在生成页面的音色下拉框里。

临时生成接口里的 `prompt_audio` 上传文件会在生成结束后删除；这里创建的自定义音色是持久保存的。

### 查询音色

```text
GET /api/voices
```

返回：

```json
{
  "voices": [
    {
      "id": "demo-1",
      "name": "示例音色",
      "title": "示例音色",
      "prompt_speech": "assets/audio/zh_1.wav",
      "audio_path": "assets/audio/zh_1.wav",
      "text": "默认合成文本",
      "language": "zh",
      "is_custom": false,
      "builtin": true,
      "created_at": "",
      "updated_at": ""
    }
  ],
  "total": 1,
  "custom_count": 0
}
```

调用 `/api/generate` 或 `/api/generate-stream/start` 时，把上面返回的 `id` 作为 `demo_id` 传入即可使用该音色。

### 新增自定义音色

```text
POST /api/voices
Content-Type: multipart/form-data
```

字段：

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `title` | string | 是 | 页面下拉框里显示的标题，最长 80 个字符。 |
| `language` | string | 否 | `zh`、`en`、`ja`、`ko`、`other`，默认 `zh`。 |
| `prompt_audio` | file | 是 | 作为音色参考的录音，支持常见音频后缀。 |
| `text` | string | 否 | 选中该音色时自动填入的默认合成文本。 |

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/voices \
  -F "title=我的旁白音色" \
  -F "language=zh" \
  -F "text=这是一段默认合成文本。" \
  -F "prompt_audio=@./my_voice.wav"
```

成功返回：

```json
{
  "voice": {
    "id": "custom-7f3a9c2e1b44",
    "name": "我的旁白音色",
    "language": "zh",
    "is_custom": true,
    "builtin": false
  }
}
```

### 更新自定义音色

```text
PUT /api/voices/{voice_id}
Content-Type: multipart/form-data
```

`title` 必填，`language` 和 `text` 可传新值；如果传 `prompt_audio`，会替换原来的录音文件。

```bash
curl -X PUT http://127.0.0.1:8000/api/voices/custom-7f3a9c2e1b44 \
  -F "title=我的新标题" \
  -F "language=zh" \
  -F "text=新的默认文本。"
```

更新系统内置音色会返回 HTTP `403`。

### 删除自定义音色

```text
DELETE /api/voices/{voice_id}
```

删除成功会移除元数据和对应录音文件：

```json
{
  "deleted": true,
  "voice_id": "custom-7f3a9c2e1b44",
  "voices": []
}
```

删除系统内置音色会返回 HTTP `403`。

## 参数建议

普通外部调用建议从这一组开始：

```text
max_new_frames=375
voice_clone_max_text_tokens=75
tts_max_batch_size=1
codec_max_batch_size=0
enable_text_normalization=1
enable_normalize_tts_text=1
cpu_threads=0
attn_implementation=model_default
do_sample=1
text_temperature=1.0
text_top_p=1.0
text_top_k=50
audio_temperature=0.8
audio_top_p=0.95
audio_top_k=25
audio_repetition_penalty=1.2
seed=0
```

如果你想要结果更可复现，设置固定整数 `seed`。

如果 CPU 压力太大，调低并发，或设置较小的 `cpu_threads`。当前服务内部会串行保护 CPU 推理执行，避免多个请求同时改 `torch.set_num_threads`。
