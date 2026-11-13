# LeetCV

Personal computer-vision practice site. 106 code problems are judged with the copied unit tests. Concept questions are critical thinking: save the written answer, then mark it solved. Progress is one global SQLite record.

## Run locally

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. For frontend work, run `npm run dev` in `frontend` (port 5173) and leave uvicorn on port 8000. The dev server proxies `/api`.

Progress is stored in `data/leetcv.db` unless `LEETCV_DATA` points somewhere else.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Deploy to leetcv.schmidlin.casa

The image is `ghcr.io/mschmidlin1/leetcv`. The cluster Service is `http://leetcv.leetcv.svc.cluster.local:80`. These steps are not done by the repo files alone:

1. On Mint, install a self-hosted runner for this repo only (a new `~/actions-runner-leetcv` directory, label `self-hosted`). `sudo ./svc.sh install`, `start`, and `status`. It should stay Idle.
2. GitHub → Settings → Actions → Workflow permissions → Read and write.
3. Name the Mint folder that should hold `leetcv.db`, create it, and set that path on the `hostPath` in `k8s/deployment.yaml`. The file currently says `/home/mike/leetcv-data`.
4. Push to `main` or run the Deploy workflow. `kubectl get pods -n leetcv` should show `Running` `1/1`.
5. After the first push, set the GHCR package `ghcr.io/mschmidlin1/leetcv` to Public.
6. On the existing Cloudflare tunnel, add a public hostname. Do not create a tunnel.

| Field | Value |
| --- | --- |
| Subdomain | `leetcv` |
| Domain | `schmidlin.casa` |
| Path | empty |
| Type | HTTP |
| URL | `http://leetcv.leetcv.svc.cluster.local:80` |

7. `curl -I https://leetcv.schmidlin.casa` should return 200.

Reference solutions used to generate the extra tests live in `tools/oracles/` and are not copied into the image.
