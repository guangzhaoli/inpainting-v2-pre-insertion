"""Read-only, narrowly scoped live training status and validation image viewer."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


PAGE = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>InsertFuse · Teacher v2</title><style>
body{font:15px/1.6 system-ui,sans-serif;background:#f4f6f8;color:#17202a;margin:0}main{max-width:1280px;margin:auto;padding:28px}h1{margin:0;font-size:29px}p{color:#54616f}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}.card,section{background:white;border:1px solid #dbe2e9;border-radius:16px;padding:22px;margin:18px 0}h2{font-size:21px;margin:0 0 14px}.muted{color:#687789;font-size:13px}.stats{display:grid;grid-template-columns:1fr 1fr;gap:14px}.stats b{display:block;font-size:25px}progress{width:100%;height:18px}.tag{background:#e4edf6;border-radius:20px;padding:4px 10px;font-size:12px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;text-align:left;border-bottom:1px solid #e7edf2}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}.gallery img{width:100%;height:310px;object-fit:contain;background:#f3f4f5}.gallery label{display:block;text-align:center}.toolbar{display:flex;gap:12px;flex-wrap:wrap}select{padding:9px;border:1px solid #ccd5df;border-radius:6px;background:white}a{color:#155eb8}.warning{color:#975800}code{font-size:12px;word-break:break-all}</style>
<main><h1>InsertFuse · Teacher v2</h1><p>首饰与宠物 · 分布式训练空间 · 每个任务单机 8 张 GPU</p><div id="stamp" class="muted"></div><div class="cards" id="cards"></div><section><h2>固定样本验证</h2><p class="muted">比较参考主体、场景与生成结果。DINO 相似度衡量参考一致性；背景 MAE 衡量原始输出变化，二者不能单独代表自然度。</p><div class="toolbar"><select id="domain"></select><select id="step"></select><select id="sample"></select></div><div id="empty" class="muted"></div><div class="gallery" id="gallery"></div></section></main>
<script>
let status={}, meta={};const $=x=>document.getElementById(x);const names={accessory_jewelry:'首饰 / 珠宝',animal:'宠物 / 猫狗'};
const fmt=(v,n=4)=>Number.isFinite(v)?v.toFixed(n):'—';const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function options(id,items,selected){$(id).innerHTML=items.map(([v,l])=>`<option value="${esc(v)}">${esc(l)}</option>`).join('');if(items.some(x=>x[0]===selected))$(id).value=selected;}
async function refresh(){try{status=await (await fetch('/status.json',{cache:'no-store'})).json();$('stamp').textContent='更新：'+new Date(status.updated_utc).toLocaleString();$('cards').innerHTML=Object.entries(status.jobs).map(([d,j])=>{let s=j.scalars??{};let metric=k=>s[k]?.mean_last_50;return `<div class="card"><h2>${names[d]} <span class="tag">${esc(j.status)}</span></h2><div class="muted">${esc(j.name)}</div><progress max="${j.estimated_steps}" value="${j.step}"></progress><div class="stats"><div><b>${j.step.toLocaleString()} / ${j.estimated_steps.toLocaleString()}</b>训练步数</div><div><b>${fmt(j.seconds_per_step,1)} 秒</b>最近 50 步耗时</div><div><b>${fmt(metric('loss'))}</b>训练损失</div><div><b>${fmt(j.eta_hours_excluding_future_validation,1)} 小时</b>预计剩余（不含后续验证）</div></div><table><tr><th>损失项</th><th>近期均值</th></tr>${[['Latent','train/latent_mse'],['前景 LPIPS','train/region_fg_lpips'],['身份 DINO','train/region_ref_dino'],['局部细节','train/region_detail_lpips'],['边界','train/region_boundary_l1']].map(([l,k])=>`<tr><td>${l}</td><td>${fmt(metric(k))}</td></tr>`).join('')}</table><p class="muted">已完成 ${j.validation.length} 次验证</p></div>`}).join('');const old=$('domain').value;options('domain',Object.keys(status.jobs).map(d=>[d,names[d]]),old);steps();}catch(e){$('stamp').textContent='等待状态文件更新：'+e.message;}}
function steps(){let d=$('domain').value;let records=status.jobs?.[d]?.validation??[];let old=$('step').value;options('step',records.map(r=>[String(r.step),'Step '+r.step]),old);if(!old&&records.length)$('step').value=String(records.at(-1).step);loadMeta();}
async function loadMeta(){let d=$('domain').value,s=$('step').value;if(!s){$('empty').textContent='训练达到验证步数后显示图像。';$('gallery').innerHTML='';$('sample').innerHTML='';return;}meta=await (await fetch(`/validation?domain=${d}&step=${s}`)).json();let old=$('sample').value;options('sample',meta.samples.map((r,i)=>[String(i),r.category+' / '+r.stem]),old);show();}
function show(){let r=meta.samples?.[Number($('sample').value)];if(!r)return;$('empty').textContent='同一验证样本使用固定随机种子。';let d=$('domain').value,s=$('step').value,i=$('sample').value;let columns=[['reference','参考主体'],['source','原始场景'],['source_mask_overlay','放置区域'],['output_restored_raw','原始生成'],['output','融合输出'],['composite','严格合成对照'],['historical_primary','历史 v1 最终结果'],['historical_raw','历史 v1 原始生成']];$('gallery').innerHTML=columns.filter(([k])=>r.paths[k]).map(([k,l])=>{let u=`/image?domain=${d}&step=${s}&sample=${i}&key=${k}`;return `<div><label>${l}</label><a href="${u}" target="_blank"><img src="${u}" loading="lazy"></a></div>`}).join('');}
$('domain').onchange=()=>{$('step').value='';steps()};$('step').onchange=loadMeta;$('sample').onchange=show;refresh();setInterval(refresh,30000);
</script></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--port", type=int, default=18865)
    parser.add_argument("--dashboard", action="store_true", help="Serve live curves and checkpoint case comparisons")
    args = parser.parse_args()
    if args.dashboard:
        from teacher_dashboard.server import serve
        serve(args.manifest, args.port)
        return
    run = args.manifest.resolve().parent

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    payload, kind = PAGE.encode(), "text/html; charset=utf-8"
                elif parsed.path == "/status.json":
                    status = json.loads((run / "status.json").read_text())
                    for job in status["jobs"].values():
                        job.pop("checkpoint", None)
                        job.pop("training_directory", None)
                        for record in job["validation"]:
                            record.pop("directory", None)
                    payload, kind = json.dumps(status).encode(), "application/json"
                elif parsed.path in {"/validation", "/image"}:
                    manifest = json.loads(args.manifest.read_text())
                    domain = query["domain"][0]
                    job = manifest["jobs"][domain]
                    step = int(query["step"][0])
                    candidates = sorted(
                        Path(job["expected_run_parent"]).glob(
                            f"v*/validation/step-{step:08d}/metadata.json"
                        ),
                        key=lambda p: p.stat().st_mtime,
                    )
                    if not candidates:
                        raise FileNotFoundError()
                    metadata_path = candidates[-1]
                    metadata = json.loads(metadata_path.read_text())
                    baselines = {}
                    baseline_file = run / "historical_baselines.json"
                    if baseline_file.exists():
                        baselines = json.loads(baseline_file.read_text())[
                            "domains"
                        ].get(domain, {})
                    if parsed.path == "/validation":
                        samples = []
                        for record in metadata["samples"]:
                            public = {k: record[k] for k in ["category", "stem"]}
                            public["paths"] = {k: True for k in record["paths"]}
                            baseline_key = f"{record['category']}/{record['stem']}"
                            if baseline_key in baselines:
                                public["paths"].update(
                                    historical_primary=True, historical_raw=True
                                )
                            samples.append(public)
                        payload = json.dumps({"samples": samples}).encode()
                        kind = "application/json"
                    else:
                        index = int(query["sample"][0])
                        key = query["key"][0]
                        if key not in {
                            "reference",
                            "source",
                            "source_mask_overlay",
                            "output_restored_raw",
                            "output",
                            "composite",
                            "historical_primary",
                            "historical_raw",
                        }:
                            raise ValueError("Unsupported image role")
                        sample = metadata["samples"][index]
                        allowed_root = metadata_path.parent.resolve()
                        if key.startswith("historical_"):
                            baseline = baselines[
                                f"{sample['category']}/{sample['stem']}"
                            ]
                            role = "raw" if key == "historical_raw" else "primary"
                            path = Path(baseline[role]).resolve()
                            allowed_root = (
                                Path(manifest["root"])
                                / "output/inference/testset_20260120/all_ckpts_final_architecture_8gpu/runs"
                            )
                        else:
                            path = Path(sample["paths"][key]).resolve()
                        if (
                            not path.is_relative_to(allowed_root)
                            or path.suffix != ".png"
                        ):
                            raise ValueError("Invalid image path")
                        payload, kind = path.read_bytes(), "image/png"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header(
                    "Cache-Control",
                    "no-store" if kind != "image/png" else "public, max-age=60",
                )
                self.end_headers()
                self.wfile.write(payload)
            except (ValueError, OSError, KeyError, IndexError):
                self.send_error(404)

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
