# Changelog

## v2.2.0 (2026-03-27)

### 🚀 Major Features
- **Multi-view Pipeline**: 4-view inference (front/back/left/right) for 3.4x more detail
- **API v2 Auth**: JWT login + API key authentication + per-key quota management
- **Async Generation**: Background task queue with 2 workers, submit & poll
- **Python SDK**: Full client library for all API endpoints
- **Three.js Preview**: Interactive 3D model viewer (orbit/wireframe/transparent)
- **Admin Dashboard**: Real-time stats, queue status, model history

### 🔧 New API Endpoints (v2)
- Auth: `/api/v2/register`, `/api/v2/login`, `/api/v2/keys`, `/api/v2/quota`
- Generation: `/api/v2/generate/async`, `/{id}/status`, `/{id}/result`, `/queue`, `/batch`
- Models: `/api/v2/models`, `/{id}`, `/{id}/download`, `/stats`
- Sharing: `/api/v2/share`, `/api/v2/gallery`, `/s/{id}`, `/embed/{id}`
- Tools: `/api/v2/convert`, `/api/v2/mesh-info`, `/api/v2/materials`, `/api/v2/materials/estimate`
- Events: `/api/v2/events` (SSE real-time stream)
- Webhooks: `/api/v2/webhooks` (register/list/remove)
- Ops: `/metrics` (Prometheus), `/health/detail`

### 🖥️ Web Pages
- `/preview` — Three.js 3D viewer with drag & drop
- `/dashboard` — Admin dashboard with live stats
- `/api-keys` — API key management UI
- `/gallery` — Public model gallery (via sharing)

### 🛠️ New Modules
- `api_v2.py` — API key auth + JWT + user management
- `billing.py` — Usage tracking + monthly stats
- `feishu_notifier.py` — Feishu webhook notifications
- `webhook.py` — Generic webhook with HMAC + retry
- `queue.py` — Async generation queue
- `model_store.py` — Model history + persistent storage
- `sdk.py` — Python client SDK
- `sse.py` — Server-Sent Events bus
- `middleware.py` — Request logging + rate limiting + error handling
- `converter.py` — Mesh format conversion + simplification
- `material_db.py` — 7 materials with properties + cost estimation
- `sharing.py` — Public share links + embed codes
- `export_glb.py` — GLB export with PBR materials
- `rate_limit.py` — Sliding window rate limiter
- `metrics.py` — Prometheus metrics collector
- `health.py` — System health checks
- `errors.py` — Standardized error codes
- `i18n.py` — Internationalization (en/zh/ja)

### 🔧 Improvements
- TripoSR local inference: Mac MPS compatible (scikit-image marching cubes fallback)
- Docker + docker-compose + one-click deploy script
- 23 CLI commands
- Pipeline `multi_view=True` parameter
- GLB format support in generate endpoint
- X-Request-ID + X-Response-Time-Ms headers on all responses

### 📊 Stats
- 61 API routes
- 42 Python modules
- 10,000+ lines of code
- 25 tests
- 23 CLI commands
- 4 web pages

---

## v2.1.0 (2026-03-25)
- TripoSR local weight download
- NSFW detection
- Product polish sprint

## v2.0.0 (2026-03-25)
- Launch content + HF auth + demo images
- 154 tests, 24 modules

## v1.0.0 (2026-03-24)
- Initial release: image to 3D print pipeline
