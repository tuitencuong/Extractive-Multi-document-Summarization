"""Interactive local web demo for the repository's KMeans/Centroid/MMR pipeline.



Run from the repository root with:

    python methods/main_method/interactive_pipeline_demo.py



The server uses the repository's preprocessing, Word2Vec, KMeans, CentroidBow,

MMR and position-restoration code. Keep the existing repository modules and dependencies installed; only this demo file is new.

"""



from __future__ import annotations



import argparse

import json

import math

import sys

from http.server import BaseHTTPRequestHandler, HTTPServer

from pathlib import Path

from typing import Any



def _find_repository_root() -> Path:
    """Locate the repository whether this file is in methods/main_method or run by absolute path."""
    expected = Path("methods/main_method/Kmeans_CentroidBase_MMR_SentencePosition.py")
    candidates = [Path(__file__).resolve().parent, *Path(__file__).resolve().parents,
                  Path.cwd(), *Path.cwd().parents]
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / expected).is_file() and (candidate / "utils/sentence.py").is_file():
            return candidate
    # Preserve the expected layout for a helpful import error if the repository is not found.
    return Path(__file__).resolve().parents[2]


ROOT_DIR = _find_repository_root()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))



PAGE = r'''<!doctype html>

<html lang="vi">

<head>

<meta charset="utf-8">

<meta name="viewport" content="width=device-width,initial-scale=1">

<title>Trực quan pipeline K-means · Centroid · MMR</title>

<style>

:root{--ink:#1d2942;--muted:#63708a;--line:#e2e8f2;--blue:#5268e8;--teal:#4a2f6f;--orange:#d9822b;--paper:#fff;--wash:#f5f7fc;--shadow:0 12px 34px #21335b12}*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 5% 0,#e8edff,transparent 34rem),#f5f7fc;color:var(--ink);font:15px/1.55 Inter,system-ui,"Segoe UI",sans-serif}main{width:min(1240px,100%);margin:auto;padding:24px 20px 40px}.hero{padding:28px clamp(20px,4vw,42px);border-radius:24px;background:linear-gradient(120deg,#1b284f,#455dc5);color:white;box-shadow:var(--shadow)}.eyebrow{color:#b9c7ff;font-size:11px;font-weight:800;letter-spacing:.14em}.hero h1{margin:8px 0;font-size:clamp(25px,4vw,38px);line-height:1.15}.hero p{max-width:850px;margin:10px 0 0;color:#e3e9ff}.layout{display:grid;grid-template-columns:minmax(0,1fr);gap:18px;margin-top:18px;align-items:start}.card{background:var(--paper);border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:var(--shadow)}.input-card{position:static;min-width:0}.card h2{font-size:18px;margin:0 0 8px}.help{color:var(--muted);font-size:13px;margin:0 0 14px}#docs{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}.doc{margin:4px 0;min-width:0}.doc textarea{min-height:86px}.doc label{display:block;margin-bottom:5px;font-size:12px;font-weight:800;color:#52607b}.doc textarea{width:100%;min-height:86px;resize:vertical;border:1px solid #d9e0ed;border-radius:11px;padding:10px 11px;font:13px/1.6 Inter,system-ui,sans-serif;color:var(--ink);background:#fbfcff}.doc textarea:focus,input:focus,select:focus{outline:3px solid #5268e82a;border-color:var(--blue)}.settings{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:13px 0}.setting label{display:block;font-size:11px;font-weight:800;color:var(--muted);margin-bottom:4px}.setting input,.setting select{width:100%;height:38px;padding:6px 9px;border:1px solid #d9e0ed;border-radius:9px;background:#fff;color:var(--ink);font:inherit}.wide{grid-column:1/-1}.actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:13px}button{border:1px solid #d5ddec;border-radius:10px;background:#fff;padding:9px 12px;color:#35415b;font:700 12px Inter,system-ui,sans-serif;cursor:pointer}button:hover{border-color:#9eacee;background:#f7f8ff}button.primary{background:var(--blue);border-color:var(--blue);color:#fff}button.primary:hover{background:#4157ce}button:disabled{opacity:.45;cursor:default}.hint{color:#758098;font-size:11px;margin:12px 0 0}.status{display:none;margin-top:12px;padding:10px 12px;border-radius:10px;font-size:12px}.status.error{display:block;background:#fff0ed;color:#9a3826;border:1px solid #f3c4ba}.status.info{display:block;background:#f2edf8;color:#422b63;border:1px solid #d3c6e6}.workspace{min-width:0}.empty{min-height:400px;display:grid;place-items:center;text-align:center;color:var(--muted)}.empty strong{display:block;color:var(--ink);font-size:18px;margin-bottom:6px}.tabs{display:flex;gap:7px;overflow-x:auto;padding:2px 0 12px}.tabs button{white-space:nowrap}.tabs button.active{background:#e8ecff;border-color:#b9c5ff;color:#344ab8}.stage{display:none}.stage.active{display:block}.stage-title{display:flex;gap:12px;align-items:flex-start;margin-bottom:15px}.stage-num{display:grid;place-items:center;flex:0 0 42px;height:42px;border-radius:13px;background:#edf0ff;color:#465bd0;font-size:13px;font-weight:900}.stage-title h2{margin:0;font-size:21px}.stage-title p{margin:3px 0 0;color:var(--muted);font-size:13px}.metric-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin:12px 0}.metric{padding:10px 12px;border:1px solid var(--line);border-radius:11px;background:#fafbff}.metric b{display:block;font-size:19px;font-variant-numeric:tabular-nums}.metric span{color:var(--muted);font-size:11px}.note{padding:11px 13px;border:1px solid #dde5ff;border-left:4px solid #6477ea;border-radius:9px;background:#f5f7ff;color:#3e4b6c;font-size:12px;margin:12px 0}.note.warn{border-color:#f0dfc4;border-left-color:var(--orange);background:#fff9ef;color:#76531f}.scroll{overflow:auto;border:1px solid var(--line);border-radius:11px;margin:12px 0}table{width:100%;min-width:540px;border-collapse:collapse;background:#fff;font-size:12px}th,td{padding:9px 10px;text-align:left;vertical-align:top;border-bottom:1px solid #ebeff5}thead th{position:sticky;top:0;background:#f5f7fc;color:#394660;font-size:11px}tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}td.num{white-space:nowrap;font-variant-numeric:tabular-nums}.small{font-size:11px;color:var(--muted)}.sentence{min-width:230px;max-width:430px}.chips{display:flex;flex-wrap:wrap;gap:5px}.chip{display:inline-block;padding:4px 7px;border:1px solid #dfe5ef;border-radius:7px;background:#fbfcff;font-size:11px;color:#44516c}.query{border:1px solid #cbbbe2;border-radius:11px;background:#f4f0f9;padding:12px 14px;margin:12px 0}.query b{display:block;color:#472e6b;font-size:11px;letter-spacing:.06em;margin-bottom:5px}.player{display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:9px;margin:12px 0}.player span{min-width:115px;text-align:center;color:var(--muted);font-size:12px;font-weight:700}.plot{width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:12px;background:#fff}.legend{display:flex;flex-wrap:wrap;gap:9px;margin:9px 0;color:var(--muted);font-size:11px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}.won{background:#f2eef8}.lost{background:#fff8ef}.formula{padding:12px;border:1px solid var(--line);border-radius:10px;background:#fafbfe;font:12px/1.7 ui-monospace,Consolas,monospace;overflow:auto}.details{border-top:1px solid var(--line);padding:10px 0}.details summary{cursor:pointer;color:#4659c0;font-weight:750;font-size:12px}.final-list{padding-left:22px}.final-list li{padding:9px 0;border-bottom:1px solid #edf0f5}.final-list small{display:block;color:var(--muted);margin-bottom:3px}.pill{display:inline-block;padding:3px 7px;border-radius:99px;background:#eee9f6;color:#402961;font-size:10px;font-weight:800}.muted{color:var(--muted)}
.mmr-flow{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:12px 0}
.mmr-branch{border:1px solid var(--line);border-radius:12px;padding:13px;background:#fbfcff;min-width:0}
.mmr-branch h4{margin:0 0 7px;font-size:13px}
.mmr-branch p{margin:5px 0;color:var(--muted);font-size:12px}
.mmr-branch.rel{border-top:4px solid var(--blue)}.mmr-branch.red{border-top:4px solid var(--orange)}.mmr-branch.score{border-top:4px solid var(--teal)}
.mmr-eq{margin-top:7px;padding:9px;border-radius:8px;background:#f2f5fb;font:12px/1.7 ui-monospace,Consolas,monospace;overflow-wrap:anywhere}
.max-red{background:#fff3df!important;font-weight:800;color:#895514}
.score-match{color:#402961;font-weight:800}.score-mismatch{padding:10px 12px;border:1px solid #f1c4ba;background:#fff2ef;border-radius:9px;color:#963b2b;font-size:12px;margin:8px 0}
@media(max-width:850px){.mmr-flow{grid-template-columns:1fr}}
@media(max-width:850px){main{padding:14px 10px 28px}.layout{grid-template-columns:1fr}.input-card{position:static}.doc textarea{min-height:82px}.hero{border-radius:18px}.card{padding:14px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;animation:none!important}}



.input-details>summary{display:flex;align-items:center;justify-content:space-between;gap:12px;list-style:none;cursor:pointer;padding:2px 0}
.input-details>summary::-webkit-details-marker{display:none}
.input-details>summary strong{display:block;font-size:15px;color:var(--ink)}
.input-details>summary small{display:block;font-size:11px;color:var(--muted);font-weight:500;margin-top:3px}
.details-toggle{padding:6px 9px;background:#f1f4ff;border:1px solid #dce3ff;border-radius:8px;color:#4d5ec5;font-size:10px;white-space:nowrap}
.input-details[open] .details-toggle{background:#f8f9fc;color:#68748b;border-color:#e2e7f0}
.input-details-body{padding-top:10px}
.input-run-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:12px}
.input-run-row .status{flex:1;margin:0;min-width:220px}
.settings .hint{grid-column:1/-1}

/* Một timeline toàn cục: mỗi slide là một thao tác nhỏ của thuật toán. */
.flow-journey{margin:18px 0 12px}.flow-heading{display:flex;justify-content:space-between;align-items:flex-end;gap:14px;margin:0 2px 12px}.flow-heading h2{margin:0;font-size:21px;letter-spacing:-.03em}.flow-heading p{margin:0;max-width:380px;color:var(--muted);font-size:12px}.flow-kicker{display:block;color:#6876d9;font-size:10px;font-weight:850;letter-spacing:.14em;margin-bottom:3px}
#flow-groups{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:8px;margin:0}.pills#flow-groups button{min-height:55px;display:flex;align-items:center;gap:8px;text-align:left;white-space:normal;padding:9px;border-radius:12px;background:#fff;font-size:11px;line-height:1.35}.pills#flow-groups button.on{color:#fff;background:linear-gradient(145deg,#5368e8,#414fc7);border-color:#5368e8}.flow-group-number{display:grid;place-items:center;flex:0 0 24px;width:24px;height:24px;border-radius:8px;background:#edf0ff;color:#5267e8;font-size:10px;font-weight:900}.pills#flow-groups button.on .flow-group-number{background:#ffffff2e;color:white}
.flow-controls{display:grid;grid-template-columns:auto auto auto minmax(150px,auto) 1fr;align-items:center;gap:8px;padding:10px;margin:13px 0 14px;border:1px solid var(--line);border-radius:14px;background:#ffffffdb;box-shadow:0 5px 16px #21335b0b}.flow-controls button{min-height:39px}.flow-controls #flow-play{background:var(--blue);color:white;border-color:var(--blue)}.flow-controls button:disabled{opacity:.4}.flow-controls span{justify-self:end;font-size:12px;font-weight:800;color:var(--muted);white-space:nowrap;font-variant-numeric:tabular-nums}.flow-controls progress{grid-column:1/-1;width:100%;height:7px;accent-color:var(--blue)}.flow-card{padding:clamp(14px,2.2vw,24px)}.flow-step-heading{display:flex;align-items:flex-start;gap:12px;margin-bottom:17px}.flow-step-badge{flex:0 0 44px;height:44px;display:grid;place-items:center;border-radius:13px;background:linear-gradient(145deg,#e9edff,#dce3ff);color:#495ed3;font-size:15px;font-weight:900}.flow-step-copy{min-width:0;flex:1}.flow-step-kicker{margin-bottom:4px;color:#6876d9;font-size:10px;font-weight:850;letter-spacing:.1em}.flow-step-copy h2{margin:0;font-size:clamp(19px,2.4vw,26px);letter-spacing:-.03em;line-height:1.25}.flow-step-copy p{margin:6px 0 0;color:var(--muted);font-size:13px;line-height:1.65}.flow-live{margin-top:3px;padding:5px 7px;border-radius:8px;background:#f4f0f9;border:1px solid #dbcfeb;color:#462d69;font-size:9px;font-weight:900;white-space:nowrap}.flow-strip{padding:11px;margin-bottom:16px;border:1px solid #e5eaf4;border-radius:13px;background:#f8faff}.flow-strip-head{display:flex;flex-wrap:wrap;justify-content:space-between;gap:5px;margin-bottom:8px;color:#3c4863;font-size:10px}.flow-strip-head span{color:var(--muted);font-weight:500}#flow-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(72px,1fr));gap:5px}.flow-tile{min-width:0;padding:7px;border-radius:8px;background:#aab3c1;color:white;font-size:11px;font-weight:850;line-height:1.3}.flow-tile small{display:block;margin-top:3px;font-size:9px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#ffffffdc}.flow-tile.active{outline:3px solid #f3bd57;outline-offset:1px}.flow-tile.selected{background:#492f6d}.flow-tile .flow-sub{opacity:.9}.flow-substep{display:inline-flex;align-items:center;gap:6px;margin:2px 0 12px;padding:5px 8px;border:1px solid #e0e7f4;border-radius:8px;background:#f8faff;color:#596783;font-size:11px;font-weight:750}.flow-substep b{color:#5267e8}.flow-equation{padding:12px 14px;border:1px solid #e0e7f5;border-radius:11px;background:#f8faff;font:12px/1.75 ui-monospace,Consolas,monospace;overflow-x:auto;margin:10px 0}.flow-focus{padding:12px 14px;margin:10px 0;border-left:4px solid #5268e8;border-radius:9px;background:#f3f6ff}.flow-focus.good{border-left-color:#503478;background:#f4f0f9}.flow-focus.warn{border-left-color:#d9822b;background:#fff8ee}.flow-focus b{display:block;margin-bottom:4px}.flow-result{padding:12px 14px;border:1px solid #d1c3e5;border-radius:11px;background:#f5f2f9;color:#3e285d;margin:12px 0}.flow-table-caption{display:flex;justify-content:space-between;align-items:center;gap:8px;margin:15px 0 7px}.flow-table-caption h3{font-size:14px;margin:0}.flow-table-caption span{font-size:10px;color:var(--muted)}.flow-inline-list{display:flex;flex-wrap:wrap;gap:6px}.flow-inline-list span{padding:5px 8px;border:1px solid #dce4f1;border-radius:8px;background:#fff;font-size:11px}.flow-color-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}.flow-vector{font-family:ui-monospace,Consolas,monospace;font-size:11px;overflow-wrap:anywhere}.flow-hidden-detail{margin-top:12px;border-top:1px solid var(--line);padding-top:7px}.flow-hidden-detail summary{cursor:pointer;color:#4a5dc8;font-size:12px;font-weight:800;padding:7px 0}.flow-card .note{line-height:1.65}.flow-card .scroll{max-width:100%}
@media(max-width:1050px){#flow-groups{grid-template-columns:repeat(4,minmax(0,1fr))}}
@media(max-width:650px){.flow-heading{align-items:flex-start;flex-direction:column;gap:4px}#flow-groups{grid-template-columns:repeat(2,minmax(0,1fr))}.flow-controls{grid-template-columns:1fr 1.15fr 1fr;padding:8px}.flow-controls button{padding:8px 5px;font-size:11px}.flow-controls span{grid-column:1/-1;justify-self:center}.flow-controls progress{grid-row:4}.flow-step-heading{gap:9px}.flow-step-badge{flex-basis:38px;height:38px}.flow-live{display:none}#flow-strip{grid-template-columns:repeat(3,minmax(0,1fr))}}

/* Đồ thị thuật toán: ưu tiên luồng và mối liên hệ thay vì chỉ dùng bảng. */
.algorithm-graph{display:flex;align-items:stretch;gap:7px;overflow-x:auto;padding:5px 2px 10px;margin:12px 0 14px}
.algorithm-node{flex:1 0 128px;min-width:128px;padding:11px 12px;border:1px solid #dbe3f2;border-radius:12px;background:linear-gradient(180deg,#fff,#f7f9ff);box-shadow:0 4px 12px #273d6810}
.algorithm-node .node-no{display:inline-grid;place-items:center;min-width:23px;height:23px;padding:0 5px;margin-bottom:7px;border-radius:7px;background:#e9edff;color:#4c60d4;font-size:10px;font-weight:900}
.algorithm-node b{display:block;font-size:12px;line-height:1.4;color:#263653}
.algorithm-node small{display:block;margin-top:5px;color:#64718b;font-size:11px;line-height:1.45}
.algorithm-arrow{display:flex;align-items:center;justify-content:center;flex:0 0 18px;color:#6575d8;font-size:21px;font-weight:900}
.algorithm-node.primary-node{border-color:#9fafff;background:#f0f3ff}.algorithm-node.good-node{border-color:#cdbde3;background:#f4f0f9}.algorithm-node.warn-node{border-color:#efd5ae;background:#fff8ed}
.graph-caption{color:#6a7690;font-size:11px;margin:-4px 0 10px}
.cosine-panel{border:1px solid #d8e2f3;border-radius:14px;padding:14px;margin:12px 0;background:linear-gradient(145deg,#fbfcff,#f5f8ff)}
.cosine-head{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;flex-wrap:wrap}
.cosine-score{font-size:25px;font-weight:900;letter-spacing:-.04em;color:#4b5fd0;font-variant-numeric:tabular-nums}
.cosine-verdict{display:inline-block;padding:5px 9px;border-radius:8px;font-size:11px;font-weight:850}
.cosine-verdict.pass{background:#eee9f5;color:#402961}.cosine-verdict.fail{background:#fff0e4;color:#a45d1e}
.cosine-meter{height:9px;background:#e5eaf4;border-radius:99px;overflow:hidden;margin:8px 0 5px}
.cosine-meter-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#6479eb,#573882)}
.cosine-threshold-row{position:relative;height:15px;margin:0 0 4px}
.cosine-threshold-mark{height:15px;width:2px;background:#d9822b;position:absolute;top:0}
.mmr-network{border:1px solid #dce4f1;border-radius:14px;background:#fbfcff;padding:13px;margin:13px 0}
.mmr-network-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;align-items:stretch}
.mmr-network-node{padding:11px;border:1px solid #d9e2f1;border-radius:10px;background:white;min-width:0}
.mmr-network-node b{display:block;font-size:12px;margin-bottom:5px}
.mmr-network-node p{margin:0;color:#63708a;font-size:11px;line-height:1.5;overflow-wrap:anywhere}
.mmr-network-node.qnode{border-top:4px solid #5268e8}.mmr-network-node.snode{border-top:4px solid #4a2f6f}.mmr-network-node.cnode{border-top:4px solid #d9822b}.mmr-network-node.winnode{border-top:4px solid #4a2f6f;background:#f5f2f9}
.mmr-network-arrow{text-align:center;color:#6474d7;font-weight:900;font-size:20px;padding:2px}
.mmr-bars{display:grid;gap:8px;margin:11px 0}
.mmr-bar-row{display:grid;grid-template-columns:minmax(60px,90px) minmax(60px,1fr) minmax(58px,90px);gap:9px;align-items:center;font-size:11px}
.mmr-bar-label{font-weight:850;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mmr-bar-track{height:15px;background:#e9edf5;border-radius:5px;overflow:hidden}
.mmr-bar-fill{height:100%;min-width:2px;background:linear-gradient(90deg,#8190ec,#5268e8);border-radius:5px}
.mmr-bar-row.winner .mmr-bar-fill{background:linear-gradient(90deg,#5b3b89,#4a2f6f)}.mmr-bar-row.winner .mmr-bar-label{color:#412a61}
.mmr-bar-value{text-align:right;font-variant-numeric:tabular-nums;font-weight:750}
@media(max-width:650px){.algorithm-node{flex-basis:118px;min-width:118px}.mmr-network-grid{grid-template-columns:1fr}.mmr-bar-row{grid-template-columns:55px minmax(45px,1fr) 75px}.cosine-score{font-size:22px}}
.km-graph-player{margin:12px 0 14px;padding:12px;border:1px solid #e0e7f2;border-radius:14px;background:#fbfcff}
.km-graph-toolbar{display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:8px;margin:0 0 9px}
.km-graph-toolbar button{min-height:35px;padding:7px 11px}
.km-graph-toolbar .km-graph-play{background:var(--blue);color:#fff;border-color:var(--blue)}
.km-graph-status{margin:4px 0 8px;text-align:center;font-size:12px;font-weight:800;color:#4659c0}
#km-chart-visual{min-width:0}
#km-chart-visual .plot{animation:chartFade .42s cubic-bezier(.22,.7,.25,1)}
#km-chart-visual .plot circle,#km-chart-visual .plot path,#km-chart-visual .plot line{transition:cx .45s cubic-bezier(.22,.7,.25,1),cy .45s cubic-bezier(.22,.7,.25,1),x1 .45s ease,y1 .45s ease,x2 .45s ease,y2 .45s ease,opacity .3s ease,fill .3s ease,stroke .3s ease}
.km-graph-player{scroll-margin-top:12px}
@keyframes chartFade{from{opacity:.5}to{opacity:1}}
.centroid-pipeline{margin:14px 0 18px;padding:14px;border:1px solid #e0e7f2;border-radius:14px;background:#fbfcff}
.centroid-cosine-list{display:grid;gap:10px;margin:12px 0}
.centroid-cosine-row{display:grid;grid-template-columns:minmax(70px,100px) minmax(120px,1fr) minmax(68px,90px);align-items:center;gap:9px;font-size:12px}
.centroid-cosine-label{font-weight:850;overflow-wrap:anywhere}
.centroid-cosine-track{height:13px;position:relative;overflow:visible;border-radius:99px;background:#e7ebf4}
.centroid-cosine-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#8b99f2,#5268e8)}
.centroid-cosine-fill.passed{background:linear-gradient(90deg,#855fbb,#4a2f6f)}
.centroid-cosine-fill.failed{background:linear-gradient(90deg,#f3c37e,#d9822b)}
.centroid-cosine-threshold{position:absolute;top:-4px;bottom:-4px;width:2px;background:#263653;z-index:2}
.centroid-cosine-value{text-align:right;font-weight:800;font-variant-numeric:tabular-nums;white-space:nowrap}
.mmr-state-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin:12px 0}
.mmr-state-card{min-width:0;padding:11px;border:1px solid #dfe6f2;border-radius:11px;background:#fff}
.mmr-state-card b{display:block;margin-bottom:5px;font-size:12px}
.mmr-state-card p{margin:0;color:#5f6c85;font-size:11px;line-height:1.6;overflow-wrap:anywhere}
.mmr-state-card.q{border-top:4px solid #5268e8}.mmr-state-card.s{border-top:4px solid #4a2f6f}.mmr-state-card.r{border-top:4px solid #d9822b}
.mmr-phase-title{display:flex;align-items:center;gap:9px;margin:14px 0 7px;font-size:14px;font-weight:850}
.mmr-phase-title span{display:grid;place-items:center;flex:0 0 27px;width:27px;height:27px;border-radius:8px;background:#e9edff;color:#5268e8;font-size:11px}
@media(max-width:650px){.mmr-state-row{grid-template-columns:1fr}.centroid-cosine-row{grid-template-columns:65px minmax(70px,1fr) 70px}}
/* Công thức Centroid: dùng phân số, chỉ số và ký hiệu toán học thay vì kiểu chữ mã nguồn. */
.math-equation,.formula.math-formula{
  display:block;
  font-family:"Cambria Math","STIX Two Math","Times New Roman",serif!important;
  font-size:15px!important;line-height:1.9!important;letter-spacing:.005em;color:#302f45;
  text-align:center;white-space:normal;
}
.math-equation .eq-line,.math-formula .eq-line{display:block;margin:3px 0;white-space:normal}
.math-equation i,.math-formula i{font-family:"Cambria Math","Times New Roman",serif;font-style:italic}
.math-equation .math-op,.math-formula .math-op{font-style:normal;font-weight:500}
.math-equation sub,.math-formula sub{font-size:.72em;vertical-align:-.3em}
.math-equation sup,.math-formula sup{font-size:.72em;vertical-align:.45em}
.math-frac{display:inline-grid;grid-template-rows:auto auto;vertical-align:middle;text-align:center;line-height:1.15;margin:0 .14em;white-space:normal}
.math-frac>span:first-child{display:block;padding:0 .25em .08em;border-bottom:1px solid currentColor}
.math-frac>span:last-child{display:block;padding:.08em .25em 0}
.math-piecewise{display:inline-grid;grid-template-columns:auto auto;column-gap:.38em;vertical-align:middle;text-align:left;line-height:1.45;margin-left:.2em}
.math-piecewise .brace{grid-row:1/3;align-self:center;font-size:2.3em;line-height:1;font-family:"Times New Roman",serif}

/* Bảng cosine: phân bổ cột theo tỷ lệ để các cột số không bị đẩy quá xa nhau. */
.centroid-cosine-table .scroll{width:100%;overflow-x:auto}
.centroid-cosine-table .scroll table{table-layout:fixed!important;width:100%!important;min-width:1120px!important}
.centroid-cosine-table .scroll th:nth-child(1),.centroid-cosine-table .scroll td:nth-child(1){width:4.2%!important}
.centroid-cosine-table .scroll th:nth-child(2),.centroid-cosine-table .scroll td:nth-child(2){width:4.5%!important}
.centroid-cosine-table .scroll th:nth-child(3),.centroid-cosine-table .scroll td:nth-child(3){width:21.3%!important}
.centroid-cosine-table .scroll th:nth-child(4),.centroid-cosine-table .scroll td:nth-child(4){width:32%!important}
.centroid-cosine-table .scroll th:nth-child(5),.centroid-cosine-table .scroll td:nth-child(5){width:9%!important}
.centroid-cosine-table .scroll th:nth-child(6),.centroid-cosine-table .scroll td:nth-child(6){width:7.4%!important}
.centroid-cosine-table .scroll th:nth-child(7),.centroid-cosine-table .scroll td:nth-child(7){width:7.4%!important}
.centroid-cosine-table .scroll th:nth-child(8),.centroid-cosine-table .scroll td:nth-child(8){width:8.7%!important}
.centroid-cosine-table .scroll th:nth-child(9),.centroid-cosine-table .scroll td:nth-child(9){width:5.5%!important}
.centroid-cosine-table .scroll th,.centroid-cosine-table .scroll td{padding:7px 7px!important}
.centroid-cosine-table .math-equation{font-size:12px!important;line-height:1.35!important;text-align:left}
.centroid-cosine-table .math-equation .eq-line{white-space:nowrap;margin:1px 0}
.centroid-cosine-table .scroll td:nth-child(4){padding-left:8px!important;padding-right:8px!important}
.math-piecewise .case{white-space:normal}
.math-explain{margin:7px 0 10px;color:#64708a;font-size:12px;line-height:1.65;text-align:left}
@media(max-width:650px){.math-equation,.formula.math-formula{font-size:14px!important}.math-frac{margin:0 .06em}}




/* Animation chỉ áp dụng bên trong vùng biểu đồ/sơ đồ thuật toán. Phần văn bản, bảng, nút và thanh điều hướng giữ tĩnh. */
@keyframes nodeEnter{0%{opacity:.35;transform:scale(.96)}75%{transform:scale(1.015)}100%{opacity:1;transform:scale(1)}}
@keyframes markerEnter{0%{opacity:.25;transform:scale(.4)}75%{transform:scale(1.18)}100%{opacity:1;transform:scale(1)}}
@keyframes focusPulse{0%,100%{opacity:.35}50%{opacity:.95}}
@keyframes drawLink{from{stroke-dashoffset:80;opacity:.15}to{stroke-dashoffset:0;opacity:1}}
@keyframes growBar{from{transform:scaleX(0)}to{transform:scaleX(1)}}

/* K-means: chỉ các phần tử thuộc SVG biểu đồ tọa độ được animate. */
.plot line{stroke-dasharray:5 4;stroke-dashoffset:80;animation:drawLink .75s ease-out forwards}
.plot circle,.plot path{transform-box:fill-box;transform-origin:center;animation:markerEnter .42s cubic-bezier(.2,.8,.2,1) both}
.plot .point-halo{fill:none;stroke:#f4b844;stroke-width:2.4;animation:focusPulse 1.45s ease-in-out infinite}
.plot .sentence-point{transition:r .2s,filter .2s}
.plot .sentence-point.focused{filter:drop-shadow(0 0 4px #f5b942);stroke:#f4b844!important;stroke-width:3!important}

/* Sơ đồ luồng thuật toán: chỉ node và mũi tên nằm trong vùng sơ đồ được animate. */
.algorithm-graph .algorithm-node{animation:nodeEnter .42s cubic-bezier(.2,.8,.2,1) both}
.algorithm-graph .algorithm-node:nth-of-type(1){animation-delay:.03s}
.algorithm-graph .algorithm-node:nth-of-type(2){animation-delay:.09s}
.algorithm-graph .algorithm-node:nth-of-type(3){animation-delay:.15s}
.algorithm-graph .algorithm-node:nth-of-type(4){animation-delay:.21s}
.algorithm-graph .algorithm-node:nth-of-type(5){animation-delay:.27s}
.algorithm-graph .algorithm-node:nth-of-type(6){animation-delay:.33s}
.algorithm-graph .algorithm-arrow{animation:nodeEnter .32s ease both;color:#6a7bea}

/* Biểu đồ cosine: chỉ thanh điểm và nhãn quyết định trong panel cosine chuyển động. */
.cosine-panel .cosine-meter-fill{transform-origin:left;animation:growBar .75s cubic-bezier(.2,.7,.2,1) both}
.cosine-panel .cosine-verdict{animation:nodeEnter .35s ease both}

/* Sơ đồ quan hệ MMR: chỉ node/mũi tên thuộc sơ đồ. */
.mmr-network .mmr-network-node{animation:nodeEnter .38s ease both}
.mmr-network .mmr-network-node:nth-child(1){animation-delay:.04s}
.mmr-network .mmr-network-node:nth-child(2){animation-delay:.1s}
.mmr-network .mmr-network-node:nth-child(3){animation-delay:.16s}
.mmr-network .mmr-network-arrow{animation:nodeEnter .28s ease both}

/* Biểu đồ thanh điểm MMR: chỉ thanh và hàng dữ liệu trong vùng chart. */
.mmr-bars .mmr-bar-fill{transform-origin:left;animation:growBar .75s cubic-bezier(.2,.7,.2,1) both}
.mmr-bars .mmr-bar-row{animation:nodeEnter .3s ease both}
.mmr-bars .mmr-bar-row:nth-child(2){animation-delay:.04s}
.mmr-bars .mmr-bar-row:nth-child(3){animation-delay:.08s}
.mmr-bars .mmr-bar-row:nth-child(4){animation-delay:.12s}
.mmr-bars .mmr-bar-row:nth-child(5){animation-delay:.16s}

/* Chuyển động chỉ dành cho biểu đồ/sơ đồ: văn bản, bảng, thẻ câu và điều khiển giữ tĩnh. */
.flow-tile,.flow-step-badge,.flow-live,.flow-result,.flow-equation{animation:none!important;transition:none!important}
.flow-controls button,.flow-controls select{transition:none!important}
.flow-controls progress{transition:none!important}
@media(prefers-reduced-motion:reduce){.plot line,.plot circle,.plot path,.plot .point-halo,.algorithm-graph .algorithm-node,.algorithm-graph .algorithm-arrow,.cosine-panel .cosine-meter-fill,.cosine-panel .cosine-verdict,.mmr-network .mmr-network-node,.mmr-network .mmr-network-arrow,.mmr-bars .mmr-bar-row,.mmr-bars .mmr-bar-fill{animation:none!important;transition:none!important}}


/* CHẾ ĐỘ BÀI GIẢNG: một sân khấu trực quan lớn, tập trung vào đúng thao tác đang học. */
.lecture-board{margin:0 0 20px;border:1px solid #d9e2f2;border-radius:18px;overflow:hidden;background:#fff;box-shadow:0 14px 38px #17264d12}
.lecture-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:15px 18px;background:linear-gradient(120deg,#172446,#293e79);color:#fff}
.lecture-brand{display:flex;align-items:center;gap:10px;min-width:0}.lecture-play-icon{display:grid;place-items:center;flex:0 0 34px;height:34px;border-radius:10px;background:#ffffff20;color:#fff;font-size:17px}
.lecture-brand b{display:block;font-size:13px;letter-spacing:.02em}.lecture-brand small{display:block;color:#c7d2fa;font-size:10px;margin-top:2px}
.lecture-counter{flex-shrink:0;border:1px solid #ffffff35;background:#ffffff14;border-radius:999px;padding:5px 9px;font-size:10px;font-weight:800;color:#f1f4ff}
.lecture-roadmap{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:5px;padding:12px 15px 0;background:#fbfcff}
.lecture-road{min-width:0;display:flex;align-items:center;gap:6px;border-radius:8px;padding:6px 7px;color:#8b95aa;background:#eef1f7;font-size:10px;font-weight:800}
.lecture-road i{display:grid;place-items:center;width:19px;height:19px;flex:0 0 19px;border-radius:6px;background:#fff;font-style:normal;font-size:9px}
.lecture-road.active{background:#e8ecff;color:#3e50c1;box-shadow:inset 0 0 0 1px #c4ceff}.lecture-road.done{background:#eee9f6;color:#402961}
.lecture-scene{position:relative;min-height:210px;padding:14px;background:linear-gradient(180deg,#fbfcff,#f4f7fd);overflow:hidden}
.lecture-scene:after{content:"";position:absolute;inset:0;pointer-events:none;background:radial-gradient(circle at 50% -10%,#dbe5ff80,transparent 60%)}
.lecture-scene>*{position:relative;z-index:1}.lecture-scene .plot{max-height:370px}.lecture-scene .legend{justify-content:center}
.lecture-scene-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;align-items:stretch}
.lecture-object{min-width:0;padding:14px 13px;border:1px solid #dfe6f3;border-radius:13px;background:#fff;box-shadow:0 5px 16px #20365e0a;animation:lecturePop .4s ease both}
.lecture-object .object-icon{display:grid;place-items:center;width:35px;height:35px;border-radius:10px;background:#edf0ff;color:#4b60d2;font-size:18px;margin-bottom:9px}
.lecture-object b{display:block;font-size:13px;color:#263653}.lecture-object p{font-size:12px;line-height:1.55;color:#68758d;margin:5px 0 0;overflow-wrap:anywhere}
.lecture-arrow{display:flex;align-items:center;justify-content:center;color:#7888dc;font-size:24px;font-weight:900;animation:lectureArrow 1.1s ease-in-out infinite}
.lecture-equation{margin:12px auto 0;max-width:860px;padding:11px 14px;text-align:center;background:#fff;border:1px dashed #cbd5ed;border-radius:11px;color:#263653;font:600 13px/1.7 ui-monospace,Consolas,monospace;overflow-wrap:anywhere}
.lecture-bars{display:grid;gap:9px;max-width:900px;margin:5px auto}.lecture-bar-row{display:grid;grid-template-columns:minmax(68px,125px) minmax(80px,1fr) minmax(58px,86px);align-items:center;gap:10px;font-size:12px}.lecture-bar-label{font-weight:850;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.lecture-track{height:17px;border-radius:5px;background:#e6ebf4;overflow:hidden;position:relative}.lecture-fill{height:100%;min-width:2px;border-radius:5px;background:linear-gradient(90deg,#7c8df1,#5368e8);transform-origin:left;animation:lectureGrow .75s cubic-bezier(.2,.75,.2,1) both}.lecture-fill.winner{background:linear-gradient(90deg,#6c45a2,#4a2f6f)}.lecture-fill.penalty{background:linear-gradient(90deg,#f1c17b,#d9822b)}.lecture-bar-value{text-align:right;font-variant-numeric:tabular-nums;font-weight:800;white-space:nowrap}
.lecture-cosine{max-width:800px;margin:12px auto;padding:16px;border:1px solid #dfe6f3;border-radius:14px;background:#fff}.lecture-cosine-top{display:flex;justify-content:space-between;align-items:center;gap:10px}.lecture-cosine-top strong{font-size:14px}.lecture-cosine-score{font-size:27px;letter-spacing:-.04em;font-weight:900;color:#4d60d4}.lecture-meter{height:15px;background:#e8edf5;border-radius:99px;position:relative;overflow:visible;margin:14px 0 8px}.lecture-meter-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#8090f0,#563882);animation:lectureGrow .8s ease both}.lecture-threshold{position:absolute;z-index:2;top:-6px;bottom:-6px;width:3px;background:#d9822b;box-shadow:0 0 0 3px #fff;border-radius:2px}.lecture-threshold-label{position:absolute;top:19px;transform:translateX(-50%);font-size:10px;font-weight:800;color:#a55b16;white-space:nowrap}
.lecture-caption{display:grid;grid-template-columns:40px minmax(0,1fr);gap:11px;align-items:start;padding:15px 18px;background:#fff;border-top:1px solid #e7ecf5}
.lecture-caption-icon{display:grid;place-items:center;width:38px;height:38px;border-radius:12px;background:#eee9f6;color:#472e6b;font-size:17px}
.lecture-caption strong{display:block;font-size:12px;color:#402961;letter-spacing:.05em}.lecture-caption p{margin:4px 0 0;font-size:13px;line-height:1.7;color:#34415d}.lecture-caption .caption-note{display:block;margin-top:5px;color:#8690a5;font-size:10px}
.lecture-scene .algorithm-graph{margin:4px 0}.lecture-scene .algorithm-node{min-width:104px;flex-basis:104px}.lecture-token-cloud{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;margin:10px auto;max-width:900px}.lecture-token{padding:9px 11px;border:1px solid #ccd7ff;border-radius:10px;background:#edf0ff;color:#4054c1;font-size:12px;font-weight:800;animation:lecturePop .35s ease both}.lecture-token small{display:block;color:#7180c7;font-size:10px;font-weight:600;margin-top:3px}
.lecture-state-columns{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}.lecture-state{padding:12px;border-radius:12px;border:1px solid #dce4f1;background:#fff;min-width:0}.lecture-state.q{border-top:4px solid #5268e8}.lecture-state.s{border-top:4px solid #4a2f6f}.lecture-state.r{border-top:4px solid #d9822b}.lecture-state b{display:block;font-size:12px;margin-bottom:7px}.lecture-state p{margin:0;font-size:12px;line-height:1.65;color:#5e6b83;overflow-wrap:anywhere}
.lecture-state-chip{display:inline-block;margin:3px 3px 0 0;padding:4px 7px;border-radius:7px;background:#f0f3fa;color:#47536b;font-size:10px;font-weight:800}.lecture-state-chip.winner{background:#e9e2f2;color:#402961}
@keyframes lecturePop{from{opacity:0;transform:translateY(8px) scale(.98)}to{opacity:1;transform:translateY(0) scale(1)}}
@keyframes lectureGrow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes lectureArrow{0%,100%{transform:translateX(0);opacity:.55}50%{transform:translateX(4px);opacity:1}}
@media(max-width:850px){.lecture-roadmap{grid-template-columns:repeat(3,minmax(0,1fr))}.lecture-scene-grid,.lecture-state-columns{grid-template-columns:1fr}.lecture-arrow{transform:rotate(90deg);min-height:20px;animation:none}.lecture-scene{padding:9px}.lecture-head{padding:12px}.lecture-caption{padding:12px}}
@media(max-width:560px){.lecture-road{font-size:9px}.lecture-road i{width:17px;height:17px;flex-basis:17px}.lecture-bar-row{grid-template-columns:56px minmax(45px,1fr) 68px;gap:7px;font-size:10px}.lecture-cosine-score{font-size:23px}.flow-controls{grid-template-columns:repeat(3,minmax(0,1fr))}.flow-controls .flow-speed-control{grid-column:1/-1}.flow-controls span{grid-column:1/-1}}


/* Lecture-grade, step-by-step scenes: the animation is constrained to the algorithm stage. */
.km-lesson-shell{border:1px solid #dbe4f3;border-radius:16px;background:#f8faff;padding:13px;margin:12px 0}
.plot text{paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round}
.type-km-final .plot{max-height:520px}
#km-move-stage .km-lesson-svg{min-height:300px}
.km-lesson-toolbar{display:flex;justify-content:center;align-items:center;gap:8px;flex-wrap:wrap;margin:4px 0 10px}
.km-lesson-toolbar button{min-height:37px}
.km-lesson-toolbar .km-primary{background:#5268e8;color:#fff;border-color:#5268e8}
.km-lesson-status{text-align:center;color:#4b5fd0;font-size:12px;font-weight:850;margin:7px 0}
.km-lesson-stage{background:#fff;border:1px solid #e0e7f2;border-radius:12px;overflow:hidden}
.km-lesson-svg{display:block;width:100%;height:auto;min-height:250px;background:#fff}
.km-lesson-svg .km-dot{transition:fill .25s,r .25s,opacity .25s}
.km-lesson-svg .km-current-ring{fill:none;stroke:#E6C98E;stroke-width:2;stroke-dasharray:3 3;animation:focusPulse .95s ease-in-out infinite}
.km-lesson-svg .km-distance-line{stroke-dasharray:5 4;animation:drawLink .45s ease-out both}
.km-lesson-svg .km-distance-label{font:9px ui-monospace,Consolas,monospace;fill:#68728A;paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round}
.km-lesson-svg .km-center-old{opacity:.24;stroke-dasharray:3 4}
.km-lesson-svg .km-center-new{animation:markerEnter .5s cubic-bezier(.2,.8,.2,1) both}.km-lesson-svg .km-center-new-group{opacity:.55;animation:centerSolidify 1.05s ease-out forwards}.km-lesson-svg .km-center-moving-star{filter:drop-shadow(0 1px 2px #7d87a322)}@keyframes centerSolidify{from{opacity:.45}to{opacity:1}}
.km-lesson-svg .km-center-move{stroke-dasharray:8 6;animation:drawLink .65s ease-out both}

/* K-Means: các trạng thái thay đổi có nhịp riêng, không làm toàn bộ điểm nhảy lại. */
@keyframes kmPointEnter{0%{opacity:0;transform:scale(.28)}72%{opacity:1;transform:scale(1.12)}100%{opacity:1;transform:scale(1)}}
@keyframes kmAssignPaint{0%{fill:#b8c1d0;transform:scale(1)}55%{fill:var(--cluster-color);transform:scale(1.42)}100%{fill:var(--cluster-color);transform:scale(1)}}
@keyframes kmDistanceDraw{from{stroke-dashoffset:75}to{stroke-dashoffset:0}}
@keyframes kmDistanceLabelIn{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:translateY(0)}}
.km-lesson-svg .km-dot{transform-box:fill-box;transform-origin:center;transition:fill .28s ease,opacity .28s ease,r .25s ease,filter .25s ease}
.km-lesson-svg .km-point-label{transition:opacity .28s ease,fill .2s ease;font-variant-numeric:tabular-nums}
.km-lesson-svg .km-dot.km-point-muted{opacity:.16}
.km-lesson-svg .km-point-label.km-point-label-muted{opacity:.18}
.km-lesson-svg .km-dot.km-point-current{opacity:1;filter:drop-shadow(0 1px 4px #e6a42e88)}
.km-lesson-svg .km-point-label-current{opacity:1}
.km-lesson-svg .km-point-enter{animation:kmPointEnter .42s cubic-bezier(.2,.8,.2,1) both}
.km-lesson-svg .km-assigned-now{fill:var(--cluster-color);animation:kmAssignPaint .52s cubic-bezier(.2,.8,.2,1) both}
.km-lesson-svg .km-distance-line{stroke-dasharray:7 5;stroke-dashoffset:75;animation:kmDistanceDraw .55s cubic-bezier(.2,.7,.2,1) forwards}
.km-lesson-svg .km-distance-label{transform-box:fill-box;transform-origin:center;animation:kmDistanceLabelIn .28s ease-out both}
.km-lesson-svg .km-seed-pulse{fill:none;stroke:#E6C98E;stroke-width:2;stroke-dasharray:3 4;animation:focusPulse .95s ease-in-out infinite}
.km-lesson-svg .km-center-still{animation:none}
.km-lesson-stage{transition:opacity .18s ease}
@media(prefers-reduced-motion:reduce){.km-lesson-svg .km-point-enter,.km-lesson-svg .km-assigned-now,.km-lesson-svg .km-distance-line,.km-lesson-svg .km-distance-label,.km-lesson-svg .km-seed-pulse{animation:none!important}}

.km-lesson-explain{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;align-items:center;padding:12px;margin-top:10px;border:1px solid #e0e7f4;border-radius:11px;background:#fff}
.km-lesson-explain b{display:block;margin-bottom:4px}.km-lesson-explain p{margin:0;color:#596783;font-size:12px;line-height:1.65}.km-lesson-metric{font-size:19px;font-weight:900;color:#465bd0;white-space:nowrap}
.km-legend{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:#68748b;margin:9px 2px 2px}.km-legend span{display:inline-flex;align-items:center;gap:5px}.km-legend i{display:inline-block;width:10px;height:10px;border-radius:50%;background:#adb7c9}.km-legend i.center{border-radius:1px;clip-path:polygon(50% 0,62% 35%,98% 35%,69% 57%,80% 95%,50% 72%,20% 95%,31% 57%,2% 35%,38% 35%);transform:none;background:#AAB8F0}.km-legend i.old{background:#fff;border:2px dashed #8994aa}
.centroid-angle-card{margin:13px 0;padding:13px;border:1px solid #dbe5f4;border-radius:14px;background:linear-gradient(145deg,#fbfcff,#f2f6ff)}
.centroid-angle-card h3{font-size:14px;margin:0 0 4px}.centroid-angle-card p{font-size:12px;color:#5d6a83;margin:0 0 7px}.centroid-angle-svg{width:100%;height:auto;display:block;max-height:270px;background:#fff;border:1px solid #e3eaf5;border-radius:10px}
.centroid-angle-svg .angle-vector{stroke-dasharray:190;stroke-dashoffset:190;animation:drawLink .85s ease-out forwards}.centroid-angle-svg .angle-arc{stroke-dasharray:100;stroke-dashoffset:100;animation:drawLink .7s .25s ease-out forwards}
.mmr-transfer-board{margin:14px 0;padding:14px;border:1px solid #dce5f2;border-radius:15px;background:#f8faff}
.mmr-transfer-q{padding:11px 13px;background:#eef2ff;border:1px solid #cfd8ff;border-radius:11px;margin-bottom:12px}.mmr-transfer-q b{display:block;color:#4258c6;margin-bottom:4px}.mmr-transfer-q p{margin:0;color:#35415b;font-size:12px}
.mmr-transfer-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;align-items:stretch}
.mmr-transfer-col{min-width:0;padding:11px;border:1px solid #dce4f1;border-radius:12px;background:#fff;overflow:hidden}.mmr-transfer-col h4{font-size:11px;letter-spacing:.04em;margin:0 0 8px;color:#596783}.mmr-transfer-col.s-before{border-top:4px solid #503478}.mmr-transfer-col.r-before{border-top:4px solid #d9822b}.mmr-transfer-col.s-after{border-top:4px solid #503478;background:#f5f2fa}.mmr-transfer-col.r-after{border-top:4px solid #9ba8bc}
.mmr-transfer-chip{padding:8px 9px;margin:6px 0;border:1px solid #e0e7f1;border-radius:9px;background:#f8faff;font-size:11px;line-height:1.5;overflow-wrap:anywhere}.mmr-transfer-chip small{display:block;color:#6b7890;font-size:10px;margin-top:2px}.mmr-transfer-chip.winner{border:2px solid #503478;background:#efeaf6;color:#3b2658;font-weight:850;box-shadow:0 5px 14px #5034781c}.mmr-transfer-chip.remain{border-color:#d5deec;background:#fff}.mmr-transfer-chip.arrive{animation:mmrArrive .85s cubic-bezier(.2,.8,.2,1) both}.mmr-transfer-chip.lose{opacity:.62}.mmr-transfer-arrow{text-align:center;color:#5368d9;font-size:22px;font-weight:900;margin:6px 0}.mmr-transfer-equation{margin-top:12px;padding:12px;border-radius:10px;background:#fff;border:1px solid #dfe6f2;font:12px/1.8 ui-monospace,Consolas,monospace;overflow-wrap:anywhere}
@keyframes mmrArrive{0%{opacity:.1;transform:translateY(22px) scale(.78)}65%{opacity:1;transform:translateY(-3px) scale(1.04)}100%{opacity:1;transform:translateY(0) scale(1)}}
@media(max-width:850px){.mmr-transfer-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.km-lesson-explain{grid-template-columns:1fr}.km-lesson-metric{white-space:normal}}
@media(max-width:520px){.mmr-transfer-grid{grid-template-columns:1fr}.km-lesson-svg{min-height:190px}.km-lesson-toolbar button{font-size:11px;padding:7px 8px}}
@media(prefers-reduced-motion:reduce){.km-lesson-svg .km-current-ring,.km-lesson-svg .km-distance-line,.km-lesson-svg .km-center-new,.km-lesson-svg .km-center-new-group,.km-lesson-svg .km-center-move,.centroid-angle-svg .angle-vector,.centroid-angle-svg .angle-arc,.mmr-transfer-chip.arrive{animation:none!important}}


/* Bản đồ 5 bước MMR luôn hiện trong mọi màn hình thuộc giai đoạn MMR. */
.mmr-roadmap{margin:0 0 16px;padding:13px;border:1px solid #dbe4f3;border-radius:14px;background:linear-gradient(180deg,#fff,#f8faff)}
.mmr-roadmap-head{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-bottom:10px}
.mmr-roadmap-head strong{display:block;font-size:14px;color:#263653}
.mmr-roadmap-head small{display:block;margin-top:3px;color:#68758d;font-size:11px;line-height:1.5}
.mmr-roadmap-round{padding:5px 8px;border-radius:8px;background:#edf0ff;color:#465bd0;font-size:10px;font-weight:900;white-space:nowrap}
.mmr-road-steps{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:7px}
.mmr-road-step{display:flex;gap:7px;align-items:flex-start;min-width:0;padding:9px 8px;border:1px solid #e0e6f1;border-radius:10px;background:#fff;color:#69758c}
.mmr-road-step .mmr-road-no{display:grid;place-items:center;flex:0 0 23px;width:23px;height:23px;border-radius:7px;background:#edf1f7;color:#61708a;font-size:10px;font-weight:900}
.mmr-road-step b{display:block;font-size:10px;line-height:1.4;color:#4a5670}
.mmr-road-step small{display:block;margin-top:4px;font-size:9px;line-height:1.4;color:#78849a}
.mmr-road-step.done{border-color:#d8cbe9;background:#f5f2fa}
.mmr-road-step.done .mmr-road-no{background:#e9e2f3;color:#402961}
.mmr-road-step.active{border-color:#8798f3;background:#eef1ff;box-shadow:0 0 0 2px #5268e818}
.mmr-road-step.active .mmr-road-no{background:#5268e8;color:white}
.mmr-road-step.active b{color:#354bbb}
.mmr-road-loop-note{margin:10px 1px 0;padding:9px 11px;border-left:3px solid #d9822b;border-radius:7px;background:#fff8ed;color:#775420;font-size:11px;line-height:1.6}
.mmr-road-loop-note b{color:#925f19}
.mmr-bars .mmr-bar-fill.red-fill{background:linear-gradient(90deg,#f1c17b,#d9822b)}
.mmr-red-legend{margin:7px 0 10px;padding:9px 11px;border:1px solid #f0dfc4;border-radius:9px;background:#fff9ef;color:#76531f;font-size:11px;line-height:1.55}
.mmr-loop-state{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:11px 0}
.mmr-loop-state>div{min-width:0;padding:10px 11px;border:1px solid #dce4f1;border-radius:10px;background:#fff}
.mmr-loop-state b{display:block;margin-bottom:5px;font-size:11px;color:#374560}
.mmr-loop-state p{margin:0;font-size:11px;line-height:1.55;color:#65718a;overflow-wrap:anywhere}
.mmr-loop-state .state-r{border-top:4px solid #4a2f6f}.mmr-loop-state .state-w{border-top:4px solid #5268e8}.mmr-loop-state .state-u{border-top:4px solid #d9822b}
@media(max-width:850px){.mmr-road-steps{grid-template-columns:repeat(3,minmax(0,1fr))}.mmr-loop-state{grid-template-columns:1fr}}
@media(max-width:560px){.mmr-road-steps{grid-template-columns:repeat(2,minmax(0,1fr))}.mmr-road-step{padding:8px 7px}}


/* MMR single-board view: four synchronized columns and inline live calculation. */
.mmr-live-shell{border:1px solid #d8e2f1;border-radius:15px;padding:12px;background:#fbfcff;}
.mmr-live-q{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:11px 13px;margin-bottom:11px;border:1px solid #cbd6ff;border-radius:11px;background:#eef2ff;color:#4057c8;}
.mmr-live-q b{display:block;font-size:12px;letter-spacing:.03em;}
.mmr-live-q small{display:block;margin-top:4px;color:#34415d;font-size:11px;line-height:1.5;}
.mmr-live-q>span{flex-shrink:0;padding:5px 8px;border:1px solid #d3dcff;background:#fff;border-radius:8px;font-size:10px;font-weight:850;color:#5368d8;}
.mmr-live-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;align-items:stretch;}
.mmr-live-column{min-width:0;min-height:270px;padding:10px;border:1px solid #d9e3f2;border-radius:12px;background:#fff;overflow-wrap:anywhere;}
.mmr-live-column h3{margin:0 0 9px;padding:0 0 8px;border-bottom:1px solid #e8edf5;color:#46516a;font-size:10px;line-height:1.45;letter-spacing:.035em;}
.mmr-live-column.before{border-top:4px solid #503478;}
.mmr-live-column.candidates{border-top:4px solid #d9822b;}
.mmr-live-column.after{border-top:4px solid #503478;background:#f6f3fa;}
.mmr-live-column.remaining{border-top:4px solid #9ba8bc;}
.mmr-live-card{margin:6px 0;padding:8px 9px;border:1px solid #dce5f3;border-radius:9px;background:#f8faff;color:#263653;min-width:0;transition:border-color .18s,background .18s,box-shadow .18s;}
.mmr-live-card-head{display:flex;align-items:center;justify-content:space-between;gap:6px;margin-bottom:4px;}
.mmr-live-card-head b{font-size:11px;color:#26405f;}
.mmr-live-card-head span{font-size:8px;font-weight:900;letter-spacing:.03em;color:#5368d8;}
.mmr-live-card p{margin:0;color:#61708b;font-size:10px;line-height:1.55;}
.mmr-live-card.active{border:2px solid #d9822b;background:#fff8ec;box-shadow:0 0 0 3px #f5b54425;}
.mmr-live-card.active .mmr-live-card-head b{color:#a85f19;}
.mmr-live-card.winner{border:2px solid #503478;background:#efeaf6;box-shadow:0 4px 12px #50347820;}
.mmr-live-card.winner .mmr-live-card-head b{color:#3f295e;}
.mmr-live-card.done{background:#f2f5fa;}
.mmr-live-card.active,.mmr-live-card.active.done{background:#fff8ec;}
.mmr-live-card.winner,.mmr-live-card.winner.done{background:#efeaf6;}
.mmr-live-mini-score{margin-top:6px;color:#5268e8;font-size:10px;font-weight:800;font-variant-numeric:tabular-nums;}
.mmr-live-column-note{margin:10px 1px 2px;color:#64718a;font-size:10px;line-height:1.6;}
.mmr-live-calc{margin-top:9px;padding:9px;border:1px solid #dde5f3;border-radius:9px;background:#fff;font-size:10px;line-height:1.55;}
.mmr-live-calc>b{display:block;margin-bottom:6px;font-size:9px;letter-spacing:.035em;}
.mmr-live-calc p{margin:6px 0 0;font-size:10px;line-height:1.55;}
.mmr-live-calc.rel-calc{border-left:3px solid #5268e8;}
.mmr-live-calc.red-calc{border-left:3px solid #d9822b;}
.mmr-live-calc.score-calc{border-left:3px solid #503478;}
.mmr-live-calc.winner-calc{border-left:3px solid #503478;background:#f5f2f9;}
.mmr-live-equation{padding:7px 8px;border-radius:7px;background:#f3f6fb;color:#34415d;font:10px/1.65 ui-monospace,Consolas,monospace;overflow-wrap:anywhere;}
.mmr-live-equation strong{color:#402961;}
.mmr-live-pair{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:4px;padding:5px 6px;margin:4px 0;border:1px solid #e5eaf3;border-radius:6px;background:#fbfcff;font-size:9px;}
.mmr-live-pair b{font-variant-numeric:tabular-nums;}
.mmr-live-pair small{color:#402961;font-size:8px;font-weight:900;}
.mmr-live-pair.max{border-color:#e1d7ee;background:#f1ecf7;}
.mmr-live-score{margin-top:6px;padding:7px 8px;border-radius:7px;background:#f1ecf7;color:#402961;font-size:10px;overflow-wrap:anywhere;}
.mmr-live-score b{font-variant-numeric:tabular-nums;}
.mmr-live-calc .score-warn{color:#9b5420;}
.mmr-live-note{margin-top:10px;padding:9px 11px;border:1px solid #dacfe9;border-radius:9px;background:#f5f2f9;color:#432b65;font-size:11px;line-height:1.6;}
.mmr-live-note.warn{border-color:#f0dfc4;background:#fff8ed;color:#76531f;}
.mmr-live-empty{padding:14px;border-radius:10px;background:#fff8ed;color:#8b5520;font-size:12px;}
@media(max-width:1050px){.mmr-live-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.mmr-live-column{min-height:180px;}}
@media(max-width:580px){.mmr-live-grid{grid-template-columns:1fr;}.mmr-live-q{align-items:flex-start;flex-direction:column;}.mmr-live-column{min-height:0;}}

/* Unified editorial theme inspired by the reference: white canvas, indigo linework, and muted pastel accents. */
:root{
  --ink:#21174A;
  --muted:#756D91;
  --line:#DED8F4;
  --blue:#3020B8;
  --teal:#765b9d;
  --orange:#D9A36C;
  --paper:#FFFFFF;
  --wash:#FBFAFF;
  --shadow:0 7px 22px rgba(48,32,184,.055);
  --pastel-lilac:#F0ECFF;
  --pastel-blue:#EAF2FF;
  --pastel-mint:#f0ebf6;
  --pastel-peach:#FFF1E3;
  --pastel-pink:#FBEAF1;
  --pastel-yellow:#FFF7D9;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scrollbar-color:#C7BDF3 #FBFAFF}
body{background:#fff!important;color:var(--ink);font-family:Inter,"Aptos",system-ui,"Segoe UI",sans-serif}
main{position:relative;z-index:1;width:min(1280px,100%);padding:24px clamp(12px,2.2vw,28px) 44px}
/* Fine frame from the reference poster; it never intercepts clicks. */
body::before{content:"";position:fixed;inset:4px;border:1px solid #DCD4FF;pointer-events:none;z-index:1000}
.hero{background:#fff!important;color:var(--ink)!important;border:1px solid var(--blue);border-radius:4px!important;box-shadow:none!important}
.hero h1,.hero h2{color:var(--blue)!important;font-family:"Arial Narrow","Roboto Condensed",Inter,sans-serif;letter-spacing:-.035em}
.hero p,.hero .eyebrow{color:var(--muted)!important}
h1,h2,h3,h4{color:var(--ink);font-family:"Arial Narrow","Roboto Condensed",Inter,system-ui,sans-serif;letter-spacing:-.025em}
h2{font-weight:800}
h3{font-weight:800}
.card,.flow-card,.input-card{background:var(--paper);border:1px solid var(--line);border-radius:9px;box-shadow:var(--shadow)}
.card{padding:clamp(13px,1.7vw,20px)}
.flow-card{border-top:3px solid var(--blue)}
.card h2,.stage-title h2,.flow-step-copy h2,.flow-heading h2{color:var(--blue)}
.flow-heading{padding-bottom:10px;border-bottom:1px solid #E7E1FA}
.flow-kicker,.flow-step-kicker{color:#6958C8!important;letter-spacing:.12em}
.flow-step-badge,.stage-num{background:var(--pastel-lilac)!important;color:var(--blue)!important;border:1px solid #D8D0FB;border-radius:6px!important}
.flow-live{background:var(--pastel-mint)!important;border-color:#daceea!important;color:#55397c!important;border-radius:4px!important}
.input-details>summary{padding:8px 0 12px;border-bottom:1px solid #EAE5F8}
.input-details>summary strong{color:var(--blue)!important;font-family:"Arial Narrow","Roboto Condensed",Inter,sans-serif;font-size:17px!important}
.details-toggle{background:var(--pastel-lilac)!important;border-color:#D8D0FB!important;color:var(--blue)!important;border-radius:4px!important}
.doc label,.setting label{color:#655B84!important;letter-spacing:.015em}
.doc textarea,.setting input,.setting select,input,select,textarea{border-color:#DCD5F2!important;border-radius:5px!important;background:#fff!important;color:var(--ink)!important;box-shadow:none}
.doc textarea:focus,.setting input:focus,.setting select:focus,input:focus,select:focus,textarea:focus{outline:3px solid #3020B81C!important;border-color:var(--blue)!important}
button{border:1px solid #D8D1F0;border-radius:5px;background:#fff;color:var(--blue);box-shadow:none;font-weight:750;transition:background .16s,border-color .16s,transform .16s}
button:hover{border-color:#8E7DE0;background:var(--pastel-lilac);color:#2917A4}
button:active{transform:translateY(1px)}
button.primary,.flow-controls #flow-play,.km-graph-toolbar .km-graph-play,.km-lesson-toolbar .km-primary{background:var(--blue)!important;border-color:var(--blue)!important;color:#fff!important}
button.primary:hover,.flow-controls #flow-play:hover,.km-graph-toolbar .km-graph-play:hover,.km-lesson-toolbar .km-primary:hover{background:#24158F!important;border-color:#24158F!important}
button:disabled{opacity:.42}
.actions button#sample{background:var(--pastel-blue)}
.status{border-radius:5px!important}
.status.error{background:#FFF0EE!important;border-color:#F0C9C4!important;color:#93453C!important}
.status.info{background:var(--pastel-mint)!important;border-color:#daceea!important;color:#55397c!important}
.hint,.small,.muted,.help{color:var(--muted)!important}
.tabs{gap:6px;padding-bottom:10px}
.tabs button{border-radius:4px;white-space:nowrap}
.tabs button.active{background:var(--pastel-lilac)!important;border-color:#BEB1F0!important;color:var(--blue)!important}
#flow-groups{gap:6px}
.pills#flow-groups button{border-radius:5px!important;border-color:#E1DBF4!important;background:#fff!important;color:#5D547B!important;box-shadow:none!important}
.pills#flow-groups button.on{background:var(--pastel-lilac)!important;border-color:#A99BEA!important;color:var(--blue)!important}
.flow-group-number{background:var(--pastel-blue)!important;color:var(--blue)!important;border-radius:4px!important}
.pills#flow-groups button.on .flow-group-number{background:#D9D0FF!important;color:var(--blue)!important}
.flow-controls{background:#fff!important;border:1px solid var(--line)!important;border-radius:7px!important;box-shadow:none!important}
.flow-controls progress{accent-color:var(--blue)}
.flow-controls progress::-webkit-progress-bar{background:#EEEAF8;border-radius:99px}
.flow-controls progress::-webkit-progress-value{background:var(--blue);border-radius:99px}
.flow-strip,.flow-substep{background:#FBFAFF!important;border-color:#E3DDF6!important;border-radius:6px!important}
.flow-tile{background:#E9E5F6!important;color:#51496C!important;border:1px solid #DAD3F0;border-radius:4px!important}
.flow-tile.active{outline:2px solid #E2BC86!important;outline-offset:1px}
.flow-tile.selected{background:var(--pastel-mint)!important;color:#55397c!important;border-color:#d8cbe9!important}
.flow-equation,.formula,.mmr-eq,.mmr-transfer-equation,.mmr-live-equation{background:#FBFAFF!important;border-color:#E3DDF6!important;color:#3E355F!important;border-radius:5px!important}
.flow-focus,.note,.query,.flow-result,.metric,.cosine-panel,.centroid-pipeline,.centroid-angle-card,.mmr-flow,.mmr-network,.mmr-transfer-board,.mmr-roadmap,.mmr-live-shell,.km-graph-player,.km-lesson-shell{border-radius:7px!important;box-shadow:none!important}
.flow-focus{background:var(--pastel-lilac)!important;border-color:#D8D0FB!important;border-left:3px solid var(--blue)!important;color:#3C315F!important}
.flow-focus.good,.flow-result,.query{background:var(--pastel-mint)!important;border-color:#daceea!important;color:#44315f!important}
.flow-focus.warn,.note.warn,.mmr-red-legend,.mmr-loop-state .state-u,.mmr-live-note.warn{background:var(--pastel-peach)!important;border-color:#F0D6BA!important;color:#755637!important}
.note{background:var(--pastel-blue)!important;border-color:#D9E5FB!important;border-left-color:#8DA8E8!important;color:#485574!important}
.flow-result.good{background:var(--pastel-mint)!important;border-color:#daceea!important}
.metric{background:#fff!important;border-color:var(--line)!important;border-radius:5px!important}
.metric b{color:var(--blue);font-family:"Arial Narrow","Roboto Condensed",Inter,sans-serif}
.metric span{color:var(--muted)!important}
.scroll{border-color:var(--line)!important;border-radius:6px!important}
table{background:#fff!important;border-color:var(--line)!important}
th,td{border-bottom-color:#EDE9F8!important}
thead th{background:var(--pastel-lilac)!important;color:#3B2A8D!important;font-weight:800!important}
tbody tr:hover td{background:#FCFAFF}
tbody tr.won td,tbody tr.won th{background:var(--pastel-mint)!important}
tbody tr.lost td{background:#FFFAF3!important}
.max-red{background:var(--pastel-peach)!important;color:#775637!important}
.chip,.flow-inline-list span{background:#fff!important;border-color:#DED8F4!important;border-radius:4px!important;color:#51486F!important}
.pill{background:var(--pastel-lilac)!important;color:var(--blue)!important;border:1px solid #DCD3FA!important;border-radius:4px!important}
.won{background:var(--pastel-mint)!important}
.lost{background:#FFF7EC!important}
.details{border-top-color:#E7E1F6!important}
.details summary,.flow-hidden-detail summary{color:var(--blue)!important}
.final-list li{border-bottom-color:#EDE9F8!important}
.algorithm-node{background:#fff!important;border-color:#DED8F4!important;box-shadow:none!important;border-radius:6px!important}
.algorithm-node.primary-node{background:var(--pastel-lilac)!important;border-color:#D2C8FA!important}
.algorithm-node.good-node{background:var(--pastel-mint)!important;border-color:#daceea!important}
.algorithm-node.warn-node{background:var(--pastel-peach)!important;border-color:#F0D6BA!important}
.algorithm-node .node-no{background:#E8E2FF!important;color:var(--blue)!important;border-radius:4px!important}
.algorithm-arrow,.mmr-transfer-arrow{color:#8D80D0!important}
.graph-caption{color:var(--muted)!important}
.mmr-branch,.mmr-network-node,.mmr-transfer-col,.mmr-road-step,.mmr-live-column,.mmr-loop-state>div{background:#fff!important;border-color:var(--line)!important;border-radius:6px!important}
.mmr-branch.rel,.mmr-network-node.qnode,.mmr-live-calc.rel-calc{border-top-color:#9F91E6!important;border-left-color:#9F91E6!important}
.mmr-branch.red,.mmr-network-node.cnode,.mmr-live-calc.red-calc{border-top-color:#E0B17F!important;border-left-color:#E0B17F!important}
.mmr-branch.score,.mmr-network-node.snode,.mmr-network-node.winnode,.mmr-live-calc.score-calc,.mmr-live-calc.winner-calc{border-top-color:#a58cc7!important;border-left-color:#a58cc7!important}
.mmr-network-node.winnode,.mmr-transfer-col.s-after,.mmr-live-column.after{background:var(--pastel-mint)!important}
.mmr-transfer-col.s-before,.mmr-live-column.before{border-top-color:#a58cc7!important}
.mmr-transfer-col.r-before,.mmr-live-column.candidates{border-top-color:#E0B17F!important}
.mmr-transfer-chip{border-color:#E4DFF3!important;background:#FBFAFF!important;border-radius:5px!important}
.mmr-transfer-chip.winner,.mmr-live-card.winner{background:var(--pastel-mint)!important;border-color:#b7a1d6!important;color:#44315f!important;box-shadow:none!important}
.mmr-transfer-chip.active,.mmr-live-card.active{background:var(--pastel-peach)!important;border-color:#E3BD95!important;box-shadow:none!important}
.mmr-live-q{background:var(--pastel-lilac)!important;border-color:#D8D0FB!important;color:var(--blue)!important;border-radius:5px!important}
.mmr-live-q small{color:#51486F!important}
.mmr-live-q>span{background:#fff!important;border-color:#DDD5FA!important;color:var(--blue)!important;border-radius:4px!important}
.mmr-live-card,.mmr-live-calc,.mmr-live-pair{border-color:#E5DFF4!important;background:#fff!important;border-radius:5px!important}
.mmr-live-card.active,.mmr-live-card.active.done{background:var(--pastel-peach)!important}
.mmr-live-card.winner,.mmr-live-card.winner.done{background:var(--pastel-mint)!important}
.mmr-live-score,.mmr-live-pair.max{background:var(--pastel-mint)!important;color:#44315f!important;border-color:#daceea!important}
.mmr-live-note{background:var(--pastel-mint)!important;border-color:#daceea!important;color:#44315f!important}
.mmr-road-step.active{background:var(--pastel-lilac)!important;border-color:#BEB1F0!important;box-shadow:none!important}
.mmr-road-step.active .mmr-road-no{background:#DCD3FF!important;color:var(--blue)!important}
.mmr-road-step.done{background:var(--pastel-mint)!important;border-color:#daceea!important}
.mmr-road-loop-note{background:var(--pastel-peach)!important;border-left-color:#E0B17F!important;color:#755637!important}
.mmr-bars .mmr-bar-track,.lecture-track{background:#EEEAF8!important}
.mmr-bars .mmr-bar-fill{background:linear-gradient(90deg,#C4BBF5,#8777DB)!important}
.mmr-bars .mmr-bar-row.winner .mmr-bar-fill{background:linear-gradient(90deg,#c7b5e0,#8c6fb4)!important}
.mmr-bars .mmr-bar-fill.red-fill{background:linear-gradient(90deg,#F4D6B6,#DDA66F)!important}
.cosine-meter{background:#EEEAF8!important}
.cosine-meter-fill{background:linear-gradient(90deg,#B9AFF0,#7C6DD0)!important}
.cosine-verdict.pass{background:var(--pastel-mint)!important;color:#55397c!important}
.cosine-verdict.fail{background:var(--pastel-peach)!important;color:#8C6542!important}
.centroid-cosine-track{background:#EEEAF8!important}
.centroid-cosine-fill{background:linear-gradient(90deg,#C9C0F4,#8C7ED8)!important}
.centroid-cosine-fill.passed{background:linear-gradient(90deg,#cbbae2,#9479bb)!important}
.centroid-cosine-fill.failed{background:linear-gradient(90deg,#F7DAB9,#DDA66F)!important}
.centroid-cosine-threshold{background:#6C5AC8!important}
.mmr-bar-fill{background:linear-gradient(90deg,#C9C0F4,#8C7ED8)!important}
.mmr-bar-row.winner .mmr-bar-fill{background:linear-gradient(90deg,#cbbae2,#9479bb)!important}
.mmr-bar-row .mmr-bar-label,.mmr-bar-row .mmr-bar-value{color:#51486F}
.km-graph-player,.km-lesson-shell,.centroid-pipeline,.centroid-angle-card,.mmr-transfer-board,.mmr-roadmap,.mmr-live-shell{background:#fff!important;border:1px solid var(--line)!important}
.km-lesson-stage,.centroid-angle-svg,.plot,.km-lesson-svg{background:#fff!important;border-color:#E2DCF5!important;border-radius:4px!important}
.km-lesson-toolbar,.km-graph-toolbar{gap:6px}
.km-lesson-toolbar button,.km-graph-toolbar button{border-radius:4px!important}
.km-lesson-status,.km-graph-status{color:var(--blue)!important}
.km-lesson-explain{background:#fff!important;border-color:#E3DDF6!important;border-radius:5px!important}
.km-lesson-metric,.cosine-score{color:var(--blue)!important}
.km-legend,.legend{color:var(--muted)!important}
.km-legend i.center{background:#B9AFF0!important}
.km-legend i.old{border-color:#B4A9D8!important}
.plot line{stroke-linecap:round}
.centroid-distance-chip rect{stroke-width:1!important}
.centroid-distance-chip text{fill:#30255C!important}
.centroid-angle-card{background:#fff!important}
.mmr-live-column h3,.mmr-roadmap-head strong{color:var(--blue)!important}
.score-match{color:#684c8e!important}
.score-mismatch{background:#FFF0EE!important;border-color:#F0C9C4!important;color:#93453C!important}
@media(max-width:850px){main{padding:14px 10px 32px}.card{padding:13px}.flow-card{padding:14px}}
@media(max-width:560px){body::before{inset:2px}.flow-controls{border-radius:5px!important}.pills#flow-groups button{min-height:48px}}


/* Typography fix: use fonts with reliable Vietnamese glyph coverage and improve readability. */
:root{--font-ui:"Segoe UI",Arial,sans-serif;--font-mono:Consolas,"Courier New",monospace}
html,body,button,input,select,textarea,table,th,td,
.doc label,.setting label,.hint,.status,.note,.help,.small,.chip,.pill,
.tabs button,.pills#flow-groups button,.flow-controls,.flow-controls label,
.flow-step-copy,.flow-step-copy p,.flow-strip,.flow-tile,.flow-focus,
.algorithm-node,.graph-caption,.metric,.metric span,
.lecture-road,.lecture-object,.lecture-caption,.lecture-caption p,
.km-lesson-status,.km-lesson-explain,.km-legend,.mmr-branch,.mmr-branch p,
.mmr-network-node,.mmr-network-node p,.mmr-transfer-q,.mmr-transfer-q p,
.mmr-transfer-col,.mmr-transfer-chip,.mmr-road-step,.mmr-road-step b,
.mmr-road-step small,.mmr-road-loop-note,.mmr-loop-state,.mmr-loop-state p,
.mmr-live-q,.mmr-live-q small,.mmr-live-column,.mmr-live-column-note,
.mmr-live-card,.mmr-live-card p,.mmr-live-calc,.mmr-live-note,.mmr-live-empty,
.final-list,.final-list small,.centroid-angle-card p,.cosine-panel,
.centroid-cosine-row,.centroid-cosine-label,.centroid-cosine-value {
  font-family:var(--font-ui)!important;
}
h1,h2,h3,h4,.hero h1,.hero h2,.card h2,.stage-title h2,.flow-heading h2,
.flow-step-copy h2,.input-details>summary strong,.metric b {
  font-family:var(--font-ui)!important;
}
.formula,.flow-equation,.mmr-eq,.mmr-transfer-equation,.mmr-live-equation,
.km-lesson-svg .km-distance-label,.centroid-angle-svg text {
  font-family:var(--font-mono)!important;
}
body{font-size:16px!important;line-height:1.6!important}
button{font-size:14px!important;line-height:1.35!important}
input,select,textarea,.doc textarea{font-size:14px!important;line-height:1.55!important}
.doc label,.setting label{font-size:13px!important;line-height:1.4!important}
.hint,.help,.small,.status,.note,.graph-caption,.stage-title p,
.flow-step-copy p,.lecture-caption p,.km-lesson-explain p,
.mmr-transfer-q p,.mmr-live-card p,.mmr-live-column-note,.mmr-live-note,
.mmr-road-loop-note,.score-mismatch,.mmr-empty,.query,.details summary,
.flow-hidden-detail summary{font-size:13px!important;line-height:1.6!important}
table{font-size:13px!important;line-height:1.5!important}
thead th{font-size:12px!important}
.card h2,.stage-title h2{font-size:21px!important;line-height:1.3!important}
.flow-heading h2,.flow-step-copy h2{font-size:clamp(21px,2.5vw,27px)!important;line-height:1.3!important}
.metric b{font-size:21px!important}
.metric span,.flow-controls span,.player span,.legend,.km-legend{font-size:12px!important}
.flow-kicker,.flow-step-kicker,.eyebrow,.flow-strip-head,
.lecture-caption strong,.mmr-live-column h3,.mmr-live-calc>b{font-size:11px!important}
.flow-live,.flow-group-number,.flow-tile,.flow-substep,.flow-inline-list span,
.algorithm-node .node-no,.lecture-road,.lecture-token,.lecture-state-chip,
.mmr-roadmap-round,.mmr-live-card-head span,.mmr-live-pair small{font-size:11px!important}
.algorithm-node b,.algorithm-node small,.mmr-branch h4,.mmr-network-node b,
.mmr-live-card-head b,.mmr-loop-state b,.mmr-road-step b{font-size:13px!important}
.algorithm-node small,.mmr-network-node p,.mmr-loop-state p,.mmr-road-step small{line-height:1.5!important}
.km-lesson-svg .km-distance-label{font-size:11px!important}
@media(max-width:650px){
  body{font-size:15px!important}
  button{font-size:13px!important}
  .flow-step-copy p,.note,.hint,.help{font-size:13px!important}
  .flow-controls span{font-size:12px!important}
}


/* High-contrast pass: darken text on pastel surfaces; reserve white text for dark controls only. */
:root{
  --ink:#201638;
  --muted:#514765;
}
body,main,.card,.flow-card,.input-card,.flow-content,.stage,.workspace{
  color:#201638!important;
}
.card p,.flow-card p,.input-card p,.stage-title p,.flow-step-copy p,
.flow-heading p,.flow-table-caption span,.flow-strip-head span,
.metric span,.small,.muted,.help,.hint,.graph-caption,.legend,.km-legend,
.mmr-branch p,.mmr-network-node p,.mmr-loop-state p,.mmr-road-step small,
.mmr-road-loop-note,.mmr-transfer-chip small,.mmr-transfer-col,
.mmr-live-card p,.mmr-live-column-note,.lecture-object p,.lecture-caption p,
.lecture-state p,.final-list small,.centroid-angle-card p,.flow-controls span{
  color:#514765!important;
}
.doc label,.setting label,.label,.details summary,.flow-hidden-detail summary,
.flow-substep,.flow-inline-list span,.chip,.algorithm-node small,
.algorithm-node b,.mmr-branch h4,.mmr-network-node b,.mmr-live-card-head b,
.mmr-live-calc,.mmr-loop-state b,.mmr-road-step b,.mmr-live-column h3{
  color:#30264A!important;
}
/* Sentence tiles use pastel backgrounds, so their labels must not remain white. */
.flow-tile,.flow-tile small,.flow-tile .flow-sub{
  color:#392D55!important;
  opacity:1!important;
}
.flow-tile small{font-weight:600!important;}
.flow-tile.selected,.flow-tile.selected small,.flow-tile.selected .flow-sub{
  color:#36244f!important;
}
/* Make active navigation and pastel badges dark, readable and consistent. */
.pills#flow-groups button.on,
.pills#flow-groups button.on span,
.pills#flow-groups button.on .flow-group-number,
.mmr-road-step.active,.mmr-road-step.active b,
.mmr-road-step.active small,.mmr-road-step.active .mmr-road-no{
  color:#3020A0!important;
}
.flow-live,.status.info,.flow-result,.flow-focus.good,.query,
.mmr-live-note,.mmr-live-score,.mmr-transfer-chip.winner{
  color:#392556!important;
}
.flow-focus.warn,.note.warn,.mmr-red-legend,.mmr-road-loop-note,
.mmr-live-note.warn,.score-mismatch{
  color:#70451F!important;
}
/* Chart labels remain dark even when the adjacent line or point uses a pastel color. */
.plot text,.km-lesson-svg text,.centroid-angle-svg text{
  fill:#30264A;
}
.centroid-distance-chip text{fill:#30264A!important;}
/* Keep white type only on buttons/headers whose backgrounds are deliberately dark. */
button.primary,.flow-controls #flow-play,.km-graph-toolbar .km-graph-play,
.km-lesson-toolbar .km-primary,.lecture-head,.lecture-head *{
  color:#fff!important;
}
button.primary,.flow-controls #flow-play,.km-graph-toolbar .km-graph-play,
.km-lesson-toolbar .km-primary{font-weight:800!important;}


/* Bảng câu đại diện theo từng tâm K-means */
.rep-cluster-panel{margin:16px 0;padding:15px;border:1px solid #DCD5F3;border-radius:12px;background:#fff}
.rep-cluster-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:10px}
.rep-cluster-head h3{margin:0;color:#30264A;font-size:17px}
.rep-cluster-head p{margin:5px 0 0;color:#4C4564;font-size:13px;line-height:1.5}
.rep-count{flex-shrink:0;padding:6px 9px;border:1px solid #DCD5F3;border-radius:8px;background:#F3F0FC;color:#493C77;font-size:12px;font-weight:750}
.rep-chosen-summary{display:flex;align-items:flex-start;gap:10px;padding:11px 12px;margin:10px 0;border:1px solid #c8b9dd;border-radius:9px;background:#f3f0f8;color:#2c203c}
.rep-chosen-summary .rep-star{display:grid;place-items:center;flex:0 0 28px;width:28px;height:28px;border-radius:7px;background:#e0d6ee;color:#3a2658;font-size:19px}
.rep-chosen-summary b{display:block;color:#2f1e46;font-size:14px}
.rep-chosen-summary p{margin:3px 0;color:#32293f;font-size:13px;line-height:1.5}
.rep-chosen-summary small{color:#2f1e46;font-size:12px;font-weight:750}
.rep-cluster-panel table{min-width:760px;font-size:13px}
.rep-cluster-panel th{color:#30264A;background:#F2EFFB;font-size:12px}
.rep-cluster-panel td{color:#29243C}
.rep-cluster-panel tr.rep-selected-row td{background:#ede7f5!important;border-top:1px solid #af9acd;border-bottom:1px solid #af9acd;font-weight:650}
.rep-cluster-panel tr.rep-member-row td{background:#FCFBFF}
.rep-cluster-panel tr.rep-other-row td{color:#625D70}
.rep-cluster-panel tr.rep-selected-row td:first-child{border-left:4px solid #704f9f}
.rep-cluster-panel .rep-cluster-tag,.rep-member-pill,.rep-outside-pill,.rep-selected-pill{display:inline-block;padding:4px 7px;border-radius:6px;font-size:11px;font-weight:750;white-space:nowrap}
.rep-cluster-tag{background:#F0ECFA;border:1px solid #DCD2F2;color:#40345F}
.rep-member-pill{background:#efeaf5;border:1px solid #d4c8e5;color:#3b2659}
.rep-outside-pill{background:#F6F0E8;border:1px solid #EAD9C4;color:#78572E}
.rep-selected-pill{background:#dfd5ed;border:1px solid #b9a6d4;color:#34224e}
.rep-cluster-note{margin:9px 1px 0;color:#4B3F65;font-size:12px;line-height:1.55}
@media(max-width:650px){.rep-cluster-panel{padding:10px}.rep-cluster-head{flex-direction:column}.rep-count{white-space:normal}}

</style>

</head>

<body><main>

<div class="layout"><aside class="card input-card"><details id="input-details" class="input-details" open><summary><span><strong>Văn bản đầu vào & tham số</strong><small>Mở để sửa tài liệu, số cụm K, ngưỡng cosine hoặc λ của MMR.</small></span><b class="details-toggle">Hiện / ẩn</b></summary><div class="input-details-body"><div id="docs"></div><div class="actions"><button id="add-doc" type="button">＋ Thêm tài liệu</button><button id="sample" type="button">Khôi phục mẫu</button></div>

<div class="settings"><div class="setting"><label for="clusters">Số cụm K</label><select id="clusters"><option value="custom" selected>Tùy chỉnh</option><option value="auto">Tự động theo main</option></select></div><div class="setting"><label for="custom-k">K tùy chỉnh</label><input id="custom-k" type="number" value="6" min="1" max="30"></div><div class="setting"><label for="mode">Chế độ MMR</label><select id="mode"><option value="train" selected>Train · 70% ứng viên</option><option value="test">Test · 70% ứng viên</option></select></div><div class="setting"><label for="lambda">λ trong MMR</label><input id="lambda" type="number" value="0.6" min="0" max="1" step="0.05"></div><div class="setting"><label>Chọn câu Centroid</label><div class="flow-equation" style="margin:0;padding:8px">Cosine + ngưỡng lọc trùng</div></div><div class="setting"><label for="cosine-threshold">Ngưỡng cosine</label><input id="cosine-threshold" type="number" value="0.35" min="0" max="1" step="0.05"></div><p class="hint wide">K tùy chỉnh mặc định là 8 để quan sát nhiều cụm. K-means chạy đến khi hội tụ, không cố định 2 vòng. Chỉnh ngưỡng cosine hoặc λ rồi bấm “Chạy pipeline” để cập nhật tập câu chuyển sang MMR.</p></div></div></details><div class="input-run-row"><button class="primary" id="run" type="button">Chạy pipeline</button><div class="status" id="status" role="status"></div></div></aside>

<section class="workspace">
  <div class="card empty" id="empty"><div><strong>Chưa có kết quả</strong>Sửa văn bản mẫu hoặc nhập nội dung riêng, sau đó chạy pipeline.</div></div>
  <div id="results" style="display:none">
    <section class="flow-journey"><div class="flow-heading"><div><span class="flow-kicker">CÁC BƯỚC</span><h2>Trình tự thuật toán</h2></div><p>Chọn một giai đoạn hoặc dùng Trước/Tiếp để xem lần lượt.</p></div><div class="pills" id="flow-groups"></div></section>
    <div class="flow-controls"><button id="flow-prev" type="button">← Trước</button><button id="flow-play" class="primary" type="button">▶ Tự chạy</button><button id="flow-next" type="button">Tiếp →</button><label class="flow-speed-control" for="flow-speed">Tốc độ <select id="flow-speed" aria-label="Tốc độ tự chạy"><option value="4500">Chậm</option><option value="3000" selected>Vừa</option><option value="1800">Nhanh</option><option value="900">Rất nhanh</option></select></label><span id="flow-count" aria-live="polite"></span><progress id="flow-progress" max="1" value="0" aria-label="Tiến độ mô phỏng"></progress></div>
    <section class="card flow-card" aria-live="polite">
      <div class="flow-step-heading"><div class="flow-step-badge" id="flow-badge">01</div><div class="flow-step-copy"><div class="flow-step-kicker" id="flow-kicker"></div><h2 id="flow-title"></h2><p id="flow-description"></p></div><span class="flow-live" id="flow-live">ĐANG XEM</span></div>
      <div class="flow-strip"><div class="flow-strip-head"><b>CÁC CÂU</b><span>Màu cụm · ★ câu đã chọn</span></div><div id="flow-strip"></div></div>
      <div id="flow-content"></div>
    </section>
  </div>
</section></div>

</main>

<script>

const SAMPLE=[

 // Tài liệu 1: hướng dẫn phân loại rác tại nguồn.
 'Trong những năm gần đây, lượng rác thải nhựa tại các đô thị Việt Nam ngày càng gia tăng do nhu cầu tiêu dùng của người dân.Túi ni-lông, hộp xốp và cốc nhựa dùng một lần xuất hiện phổ biến tại chợ, siêu thị và các cửa hàng thức ăn nhanh.Phần lớn rác thải nhựa sinh hoạt chưa được phân loại tại nguồn, khiến quá trình thu gom và tái chế gặp nhiều khó khăn.Một lượng lớn rác nhựa bị thải ra sông hồ và cống thoát nước, làm tăng nguy cơ ngập úng khi mưa lớn.Thực trạng này cho thấy việc giảm thiểu rác thải nhựa cần sự tham gia của cả người dân, doanh nghiệp và cơ quan quản lý.',

 // Tài liệu 2: thu gom và xử lý vật liệu.
 'Rác thải nhựa gây ra nhiều vấn đề môi trường vì phần lớn vật liệu nhựa cần thời gian rất dài để phân hủy tự nhiên.Khi tích tụ trong môi trường nước, các mảnh nhựa có thể gây nguy hiểm cho sinh vật và làm suy giảm chất lượng hệ sinh thái.Quá trình phân rã nhựa tạo ra các hạt vi nhựa có khả năng đi vào chuỗi thức ăn và đặt ra những lo ngại về sức khỏe con người.Tại đô thị, rác nhựa làm mất mỹ quan đường phố và gia tăng áp lực lên hệ thống xử lý chất thải.Nếu không có biện pháp kiểm soát hiệu quả, chi phí khắc phục ô nhiễm nhựa có thể tiếp tục tăng trong tương lai.',

 // Tài liệu 3: kết quả thí điểm và cải tiến chương trình.
 'Một trong những giải pháp quan trọng là khuyến khích người dân sử dụng túi vải, bình nước cá nhân và các sản phẩm có thể tái sử dụng.Chính quyền đô thị cần tăng cường phân loại rác tại nguồn, cải thiện hệ thống thu gom và mở rộng các chương trình tái chế nhựa.Các doanh nghiệp có thể góp phần giảm ô nhiễm bằng cách hạn chế bao bì nhựa dùng một lần và phát triển những vật liệu thân thiện với môi trường.Hoạt động tuyên truyền tại trường học và khu dân cư giúp nâng cao nhận thức về tác hại của rác thải nhựa đối với môi trường.Sự phối hợp giữa chính quyền, doanh nghiệp và cộng đồng sẽ tạo điều kiện xây dựng đô thị sạch hơn và phát triển bền vững.'

];

const LABELS=['1 · Tiền xử lý','2 · K-means','3 · Đại diện & Centroid','4 · MMR','5 · Kết quả'];
const FLOW_GROUPS=['Đầu vào & biểu diễn','K-means','Câu đại diện','Centroid','MMR','Kết quả'];

const COLORS=['#8E9BE5','#8665b4','#D6A06E','#A78CD3','#C77F9B','#6BA9D0','#9FAB59','#8997A9','#BC8266','#8766b5'];
const KM_COLORS=['#8798E0','#9879c4','#E0A477','#B09AD8','#D991AA','#82B9D8','#B4C66F','#9BAABD','#C78B70','#9979c5'];
function starPath(cx,cy,outer=8,inner=3.6){const pts=[];for(let i=0;i<10;i++){const a=-Math.PI/2+i*Math.PI/5,r=i%2===0?outer:inner;pts.push((cx+Math.cos(a)*r).toFixed(2)+' '+(cy+Math.sin(a)*r).toFixed(2));}return 'M '+pts.join(' L ')+' Z';}

const $=id=>document.getElementById(id);let DATA=null,RUN_DOCS=[],stage=0,kmRestart=0,kmStep=0,centroidStep=0,mmrStep=0,flowSteps=[],flowIndex=0,flowTimer=null,kmAnimIndex=0,kmAnimTimer=null,kmLessonTimer=null,kmLessonState=null,kmInitTimer=null,kmInitState=null,kmMoveTimer=null,kmMoveState=null;

function esc(s){return String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

function docs(){return Array.from(document.querySelectorAll('.doc textarea')).map(x=>x.value);}

function addDoc(value){const box=document.createElement('div');box.className='doc';const n=$('docs').children.length+1;box.innerHTML='<label>TÀI LIỆU '+n+'</label><textarea spellcheck="false"></textarea>';box.querySelector('textarea').value=value||'';$('docs').appendChild(box);}

function resetSample(){ $('docs').innerHTML='';SAMPLE.forEach(addDoc); }

resetSample();$('add-doc').onclick=()=>addDoc('');$('sample').onclick=resetSample;

function status(msg,type){$('status').textContent=msg;$('status').className='status '+type;}

function num(v,d=4){if(v===null||v===undefined)return '—';if(typeof v==='string')return v;return Number.isFinite(v)?v.toFixed(d):String(v);}

function cardTitle(n,title,desc){return '<div class="stage-title"><div class="stage-num">'+n+'</div><div><h2>'+title+'</h2><p>'+desc+'</p></div></div>';}

function metrics(items){return '<div class="metric-row">'+items.map(x=>'<div class="metric"><b>'+esc(x[0])+'</b><span>'+esc(x[1])+'</span></div>').join('')+'</div>';}
// Keep a 1:1 visual scale for PCA coordinates. PCA still cannot preserve all 256-D distances.
function pcaProjector(xs,ys,left,top,width,height){
  const finiteX=xs.map(Number).filter(Number.isFinite),finiteY=ys.map(Number).filter(Number.isFinite);
  let xmin=finiteX.length?Math.min(...finiteX):-1,xmax=finiteX.length?Math.max(...finiteX):1;
  let ymin=finiteY.length?Math.min(...finiteY):-1,ymax=finiteY.length?Math.max(...finiteY):1;
  if(xmax-xmin<1e-8){xmin-=.5;xmax+=.5;}if(ymax-ymin<1e-8){ymin-=.5;ymax+=.5;}
  const spanX=xmax-xmin,spanY=ymax-ymin,padX=spanX*.10,padY=spanY*.10;
  xmin-=padX;xmax+=padX;ymin-=padY;ymax+=padY;
  const scale=Math.min(width/(xmax-xmin),height/(ymax-ymin));
  const midX=(xmin+xmax)/2,midY=(ymin+ymax)/2;
  return {x:v=>left+width/2+(Number(v)-midX)*scale,y:v=>top+height/2-(Number(v)-midY)*scale};
}
function table(headers,rows){return '<div class="scroll"><table><thead><tr>'+headers.map(h=>'<th>'+h+'</th>').join('')+'</tr></thead><tbody>'+rows.join('')+'</tbody></table></div>';}

function chips(words){return '<div class="chips">'+words.map(x=>'<span class="chip">'+esc(x)+'</span>').join('')+'</div>';}

function sentenceTable(items){return table(['ID','Tài liệu','Câu gốc','Token sau tiền xử lý'],items.map(s=>'<tr><td class="num">'+esc(s.id)+'</td><td>'+esc(s.doc_label)+'</td><td class="sentence">'+esc(s.text)+'</td><td>'+chips(s.tokens)+'</td></tr>'));}

function preprocessing(){const s=DATA.sentences;return cardTitle('01','Tách câu và tiền xử lý','Tương ứng với Preprocessing.processFilev2: underthesea tách câu, chuyển thường, tách từ, bỏ dấu câu và câu có không quá 4 token.')+metrics([[s.length,'câu hợp lệ'],[DATA.documents.length,'tài liệu'],[DATA.dropped_count,'câu bị bỏ']])+sentenceTable(s)+ (DATA.boundaries.length?'<div class="note">Chỉ số bắt đầu tài liệu (last_indexs): '+DATA.boundaries.map(x=>esc(x)).join(', ')+'</div>':'');}

function plot(frame, focusIndex=null){
  const pts=DATA.sentences,cs=frame.centers||[],after=frame.centers_after||cs;
  let xs=pts.map(p=>p.x).concat(cs.map(c=>c.x),after.map(c=>c.x)),ys=pts.map(p=>p.y).concat(cs.map(c=>c.y),after.map(c=>c.y));
  const project=pcaProjector(xs,ys,35,35,620,310),X=project.x,Y=project.y;
  const pointXY=pts.map(p=>({x:X(p.x),y:Y(p.y),id:p.id}));
  const centerXY=cs.map((c,j)=>({x:X(c.x),y:Y(c.y),id:String(j+1)}));
  const placedLabels=[];
  const offsets=[
    {dx:11,dy:-9,a:'start'},{dx:11,dy:14,a:'start'},
    {dx:-11,dy:-9,a:'end'},{dx:-11,dy:14,a:'end'},
    {dx:0,dy:-14,a:'middle'},{dx:0,dy:18,a:'middle'},
    {dx:16,dy:-16,a:'start'},{dx:16,dy:18,a:'start'},
    {dx:-16,dy:-16,a:'end'},{dx:-16,dy:18,a:'end'},
    {dx:0,dy:-22,a:'middle'},{dx:0,dy:25,a:'middle'}
  ];
  const rectFor=(x,y,text,anchor)=>{
    const w=Math.max(14,String(text).length*5.7),left=anchor==='end'?x-w:(anchor==='middle'?x-w/2:x);
    return {left,top:y-8,right:left+w,bottom:y+3};
  };
  const overlap=(a,b)=>!(a.right+2<b.left||b.right+2<a.left||a.bottom+2<b.top||b.bottom+2<a.top);
  const pickLabel=(x,y,text,ownIndex,isCenter=false)=>{
    let best=null,bestScore=Infinity;
    for(let i=0;i<offsets.length;i++){
      const o=offsets[i],lx=x+o.dx,ly=y+o.dy,box=rectFor(lx,ly,text,o.a);let score=i*.001;
      for(const r of placedLabels)if(overlap(box,r))score+=1000;
      const markers=pointXY.concat(centerXY);
      markers.forEach((m,mi)=>{
        if(!isCenter&&mi===ownIndex)return;
        if(isCenter&&mi===pointXY.length+ownIndex)return;
        const nearestX=Math.max(box.left,Math.min(m.x,box.right)),nearestY=Math.max(box.top,Math.min(m.y,box.bottom));
        const d=Math.hypot(nearestX-m.x,nearestY-m.y);
        if(d<7)score+=(7-d)*5;
      });
      if(score<bestScore){bestScore=score;best={x:lx,y:ly,anchor:o.a,box};}
    }
    placedLabels.push(best.box);return best;
  };
  let out='<svg class="plot" viewBox="0 0 700 390" role="img" aria-label="PCA projection of sentence vectors and K-means centers">';
  out+='<text x="35" y="23" font-size="12" fill="#63708a">PCA 2D: độ dài trong hình chiếu · thuật toán dùng khoảng cách vector Word2Vec 256D</text>';
  out+='<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#65708a"/></marker></defs>';
  for(let j=0;j<cs.length;j++){
    const a=cs[j],b=after[j]||a,color=KM_COLORS[j%KM_COLORS.length];
    if(Math.abs(a.x-b.x)+Math.abs(a.y-b.y)>1e-8)out+='<line x1="'+X(a.x)+'" y1="'+Y(a.y)+'" x2="'+X(b.x)+'" y2="'+Y(b.y)+'" stroke="'+color+'" stroke-width="1.4" stroke-dasharray="4 4" opacity=".62" marker-end="url(#arrow)"/>';
  }
  // Keep the nearest-center relationship visible, but downplay competing centers.
  for(const p of pts){
    const cl=frame.assignments[p.index],color=KM_COLORS[cl%KM_COLORS.length],ds=(frame.distance_matrix||[])[p.index]||[],focused=focusIndex!==null&&p.index===focusIndex;
    for(let j=0;j<cs.length;j++){
      const center=cs[j],isChosen=j===cl;
      if(center)out+='<line x1="'+X(p.x)+'" y1="'+Y(p.y)+'" x2="'+X(center.x)+'" y2="'+Y(center.y)+'" stroke="'+(isChosen?color:'#aeb8ca')+'" stroke-width="'+(focused?(isChosen?'2.2':'1.1'):(isChosen?'1.35':'.7'))+'" stroke-dasharray="'+(isChosen?'3 3':'2 5')+'" opacity="'+(focused?(isChosen?'.9':'.28'):(isChosen?'.7':'.10') )+'"><title>'+esc(p.id+' → tâm '+(j+1)+' · d² thật 256D = '+(ds.length?num(ds[j],8):'không lưu')+'; không phải độ dài trên PCA'+(isChosen?' · tâm được chọn theo 256D':' · ứng viên so sánh'))+'</title></line>';
    }
  }
  // Draw markers first so labels can be placed with collision-aware offsets around all nearby points.
  pts.forEach((p)=>{
    const cl=frame.assignments[p.index],color=KM_COLORS[cl%KM_COLORS.length],focused=focusIndex!==null&&p.index===focusIndex;
    if(focused)out+='<circle class="point-halo" cx="'+X(p.x)+'" cy="'+Y(p.y)+'" r="10"></circle>';
    out+='<circle class="sentence-point '+(focused?'focused':'')+'" cx="'+X(p.x)+'" cy="'+Y(p.y)+'" r="'+(focused?'6':'4.4')+'" fill="'+color+'" stroke="#fff" stroke-width="2"><title>'+esc(p.id+' · được gán vào cụm '+(cl+1)+' · '+p.text)+'</title></circle>';
  });
  pts.forEach((p)=>{
    const xy={x:X(p.x),y:Y(p.y)},label=pickLabel(xy.x,xy.y,p.id,p.index,false),focused=focusIndex!==null&&p.index===focusIndex;
    out+='<text x="'+label.x+'" y="'+label.y+'" text-anchor="'+label.anchor+'" font-size="'+(focused?'10':'9')+'" font-weight="'+(focused?'900':'650')+'" fill="'+(focused?'#76521D':'#283452')+'" paint-order="stroke" stroke="#fff" stroke-width="3" stroke-linejoin="round">'+esc(p.id)+'</text>';
  });
  for(let j=0;j<cs.length;j++){
    const a=cs[j],b=after[j]||a,x=X(a.x),y=Y(a.y),xx=X(b.x),yy=Y(b.y),color=KM_COLORS[j%KM_COLORS.length];
    out+='<circle cx="'+x+'" cy="'+y+'" r="3.8" fill="#fff" stroke="'+color+'" stroke-width="1.6"><title>Tâm đầu vòng '+(j+1)+'</title></circle>';
    out+='<path d="'+starPath(xx,yy,8,3.5)+'" fill="'+color+'" stroke="#fff" stroke-width="1.2"><title>Tâm sau cập nhật '+(j+1)+'</title></path>';
  }
  cs.forEach((a,j)=>{
    const b=after[j]||a,x=X(b.x),y=Y(b.y),label=pickLabel(x,y,String(j+1),j,true),color=KM_COLORS[j%KM_COLORS.length];
    out+='<text x="'+label.x+'" y="'+label.y+'" text-anchor="'+label.anchor+'" font-size="11" font-weight="900" fill="#202B50" paint-order="stroke" stroke="#fff" stroke-width="3.5" stroke-linejoin="round">'+(j+1)+'</text>';
  });
  return out+'</svg><div class="legend">'+DATA.colors.map((x,i)=>'<span><i class="dot" style="background:'+KM_COLORS[i%KM_COLORS.length]+'"></i>Cụm '+(i+1)+'</span>').join('')+'<span>nét mờ: tâm được so sánh</span><span>nét màu: tâm gần nhất được chọn</span><span>○ tâm trước vòng</span><span>★ tâm mới</span><span>⇢ hướng dịch tâm</span></div>';
}

function kmeansFinalConnections(){
  const runs=DATA.kmeans.restarts||[],best=runs.find(x=>x.best)||runs[0],frames=best?.history||DATA.kmeans.history||[],frame=frames[frames.length-1];
  if(!frame)return '<div class="note">Chưa có trạng thái cuối của K-means.</div>';
  const pts=DATA.sentences,cs=frame.centers_after||frame.centers||[];
  let xs=pts.map(p=>p.x).concat(cs.map(c=>c.x)),ys=pts.map(p=>p.y).concat(cs.map(c=>c.y));
  const project=pcaProjector(xs,ys,38,35,620,310),X=project.x,Y=project.y;
  let svg='<svg class="plot" viewBox="0 0 700 390" role="img" aria-label="K-means cuối cùng: nối từng điểm tới tâm cụm được gán">';
  for(const p of pts){const cl=(frame.post_assignments||frame.assignments||[])[p.index]??0,c=cs[cl];if(!c)continue;const color=KM_COLORS[cl%KM_COLORS.length];svg+='<line x1="'+X(p.x)+'" y1="'+Y(p.y)+'" x2="'+X(c.x)+'" y2="'+Y(c.y)+'" stroke="'+color+'" stroke-width="1.25" opacity=".68"><title>'+esc(p.id+' → tâm '+(cl+1))+'</title></line>';}
  for(const p of pts){const cl=(frame.post_assignments||frame.assignments||[])[p.index]??0,color=KM_COLORS[cl%KM_COLORS.length];svg+='<circle cx="'+X(p.x)+'" cy="'+Y(p.y)+'" r="4" fill="'+color+'" stroke="#fff" stroke-width="1.1"><title>'+esc(p.id+' · cụm '+(cl+1))+'</title></circle>';}
  cs.forEach((c,j)=>{const x=X(c.x),y=Y(c.y),color=KM_COLORS[j%KM_COLORS.length];svg+='<path d="'+starPath(x,y,8.5,3.7)+'" fill="'+color+'" stroke="#fff" stroke-width="1.2"/><text x="'+(x+10)+'" y="'+(y-8)+'" fill="#68728A" font-size="10" font-weight="800">'+(j+1)+'</text>';});
  return '<section class="centroid-pipeline"><h3>Ảnh cuối K-means · nối tâm với tất cả điểm thuộc cụm</h3><p class="small">Đường nối minh họa quan hệ điểm–tâm trong PCA 2D; việc gán cụm vẫn dựa trên khoảng cách thật của vector Word2Vec 256D.</p>'+svg+'</svg><div class="legend"><span>● Điểm câu</span><span>★ Tâm cụm cuối</span><span>Đường nối = điểm → tâm được gán</span></div></section>';
}

function kmeans(){
  const allRuns=DATA.kmeans.restarts||[];
  const activeRun=allRuns[kmRestart]||{history:DATA.kmeans.history,run:DATA.kmeans.best_run||1,best:true,final_inertia:null,seed_indices:[],trace_matches:DATA.kmeans.trace_matches_sklearn,has_full_distances:true,has_full_vectors:true};
  const h=activeRun.history||DATA.kmeans.history;
  if(kmStep>=h.length)kmStep=Math.max(0,h.length-1);
  const frame=h[kmStep],isInit=kmStep===0;
  const distanceMatrix=frame.distance_matrix||[];
  const centerHeaders=Array.from({length:DATA.kmeans.k},(_,j)=>'d² tới μ'+(j+1));
  const rows=DATA.sentences.map(s=>{
    const ds=distanceMatrix[s.index]||[];
    const dCells=centerHeaders.map((_,j)=>ds.length?'<td class="num '+(j===frame.assignments[s.index]?'max-red':'')+'">'+num(ds[j],5)+'</td>':'<td class="num">—</td>').join('');
    const assigned=frame.assignments[s.index];
    return '<tr><td><b>'+esc(s.id)+'</b></td><td class="sentence">'+esc(s.text)+'</td>'+dCells+'<td><span class="pill" style="background:'+COLORS[assigned%COLORS.length]+'20;color:'+COLORS[assigned%COLORS.length]+'">Cụm '+(assigned+1)+'</span></td><td class="num">'+num(frame.distances[s.index],5)+'</td></tr>';
  });
  const clusterRows=(frame.cluster_updates||[]).map(u=>{
    const moved=(u.relocated||[]).map(r=>(DATA.sentences[r.sample_index]?.id||('C'+(r.sample_index+1)))+' → Cụm '+(r.to_cluster+1)).join(', ');
    return '<tr><td><span class="pill" style="background:'+COLORS[u.cluster%COLORS.length]+'20;color:'+COLORS[u.cluster%COLORS.length]+'">Cụm '+(u.cluster+1)+'</span></td><td class="num">'+u.count+'</td><td>'+((u.members||[]).map(i=>DATA.sentences[i]?.id||('C'+(i+1))).join(', ')||'—')+(moved?'<br><span class="small">Chuyển cụm rỗng: '+esc(moved)+'</span>':'')+'</td><td class="num">'+num(u.shift_sq,7)+'</td><td class="small">['+(u.old_preview||[]).map(v=>num(v,4)).join(', ')+(u.old_preview?.length===8?', …':'')+']</td><td class="small">['+(u.new_preview||[]).map(v=>num(v,4)).join(', ')+(u.new_preview?.length===8?', …':'')+']</td></tr>';
  });
  const historyRows=h.slice(1).map((f,i)=>'<tr class="'+(i+1===kmStep?'won':'')+'"><td>Vòng '+f.iteration+'</td><td class="num">'+num(f.shift,7)+'</td><td class="num">'+num(f.inertia,6)+'</td><td>'+esc(f.convergence_reason||'—')+'</td><td>'+(f.converged?'<span class="pill">Đã hội tụ</span>':'Đang tiếp tục')+'</td></tr>');
  const traceNote='Chế độ K: '+DATA.kmeans.k_mode+' · lần khởi tạo '+activeRun.run+'/'+(allRuns.length||DATA.kmeans.n_init)+(activeRun.best?' (thắng theo inertia thực tế của sklearn)':'')+'. Tâm khởi tạo được lấy từ: '+(activeRun.seed_indices||[]).map(i=>DATA.sentences[i]?.id||('C'+(i+1))).join(', ')+'. '+(activeRun.trace_matches===false?'CẢNH BÁO: trace từng vòng không khớp hoàn toàn KMeans.fit ở tình huống số học đặc biệt; inertia thực tế của sklearn vẫn quyết định lần thắng.':'Trace khớp nhãn/tâm cuối của KMeans.fit.')+(activeRun.has_full_distances===false?' Do giới hạn kích thước dữ liệu, ma trận khoảng cách đầy đủ chỉ được lưu cho lần thắng; các lần còn lại vẫn có nhãn, cập nhật tâm và inertia.':'')+(activeRun.has_full_vectors===false?' Vector 256 chiều đầy đủ chỉ được lưu cho lần thắng; các lần khác vẫn hiện 8 chiều đầu và toàn bộ kết quả dịch tâm.':'');
  const updateContent=isInit?'<div class="note">Khởi tạo K-means++: chọn K tâm ban đầu. Chưa cập nhật tâm; bảng khoảng cách bên dưới cho thấy từng câu được gán vào tâm nào trong trạng thái khởi tạo.</div>':
    '<h3>Phép cập nhật tâm ở vòng '+frame.iteration+'</h3><div class="formula">Với mỗi cụm j: μⱼ mới = (Σ vector của các câu trong cụm j) / số câu dùng để cập nhật.<br>Mỗi vector câu là tổng vector Word2Vec 256 chiều. Bảng tóm tắt cho xem 8 chiều đầu; có thể mở phần bên dưới để xem mọi chiều khi dữ liệu chi tiết đã được lưu.</div>'+table(['Cụm','Số mẫu dùng khi cập nhật','Các mẫu tham gia cập nhật','Tổng dịch tâm² của cụm','8 chiều đầu của tâm cũ','8 chiều đầu của tâm mới'],clusterRows)+
    '<div class="metric-row"><div class="metric"><b>'+num(frame.shift,8)+'</b><span>Tổng dịch tâm² = Σⱼ ||μⱼ mới − μⱼ cũ||²</span></div><div class="metric"><b>'+num(frame.inertia,6)+'</b><span>Inertia sau cập nhật và gán gần nhất</span></div><div class="metric"><b>'+(frame.converged?'Có':'Chưa')+'</b><span>Hội tụ ở vòng này?</span></div></div>'+
    (activeRun.has_full_vectors&&clusterRows.length?'<details class="details"><summary>Xem đủ 256 chiều của từng tâm cũ/tâm mới (tất cả cụm)</summary>'+frame.cluster_updates.map(u=>'<details class="details"><summary>Cụm '+(u.cluster+1)+' · dịch tâm² = '+num(u.shift_sq,8)+'</summary>'+table(['Chiều','μ cũ','μ mới','Δ = mới − cũ','Δ²'],(u.old_vector||[]).map((v,i)=>'<tr><td>'+i+'</td><td class="num">'+num(v,8)+'</td><td class="num">'+num(u.new_vector?.[i],8)+'</td><td class="num">'+num(Number(u.new_vector?.[i])-Number(v),8)+'</td><td class="num">'+num((Number(u.new_vector?.[i])-Number(v))**2,10)+'</td></tr>'))+'</details>').join('')+'</details>':'<div class="note">'+(activeRun.has_full_vectors===false?'Trace lớn nên chỉ lưu trước 8 chiều của các tâm để giữ giao diện hoạt động ổn định.':'Không có vector cập nhật ở bước khởi tạo.')+'</div>');
  const summary=frame.converged?'<div class="winner-box"><b>✓ Đã hội tụ ở vòng '+frame.iteration+'.</b> '+esc(frame.convergence_reason||'')+' Kết quả cuối lấy từ KMeans.fit() của scikit-learn.</div>':'<div class="note">'+esc(frame.convergence_reason||'Chưa đạt điều kiện dừng.')+'</div>';
  const restartButtons=(allRuns.length?allRuns:[activeRun]).map((r,i)=>'<button type="button" data-km-restart="'+i+'" class="'+(i===kmRestart?'current':(r.best?'done':''))+'">Lần '+r.run+' · inertia '+num(r.final_inertia,4)+(r.best?' ★ tốt nhất':'')+(r.trace_matches===false?' ⚠ trace lệch':'')+'</button>').join('');
  const jumpButtons=h.map((f,i)=>'<button type="button" data-km-jump="'+i+'" class="'+(i===kmStep?'current':(i<kmStep?'done':''))+'">'+(i===0?'Khởi tạo':'Vòng '+f.iteration)+'</button>').join('');
  return cardTitle('02','Word2Vec và K-means · xem đầy đủ từng vòng','Một vòng Lloyd gồm gán câu theo khoảng cách đến tất cả các tâm, sau đó cập nhật các tâm bằng trung bình vector. Có thể xem tất cả lần khởi tạo, nhảy đến từng vòng và mở bảng tổng hợp để lần theo toàn bộ quá trình.')+
    metrics([[DATA.kmeans.k,'số cụm K'],[allRuns.length||DATA.kmeans.n_init,'lần khởi tạo'],[h.length-1,'vòng Lloyd trong lần này'],[num(frame.inertia,5),'inertia trạng thái này']])+ 
    '<h3>Chọn lần khởi tạo K-means++</h3><p class="help">Mỗi nút là một lần khởi tạo riêng. ★ được chọn theo inertia cuối thực tế mà sklearn trả về.</p><div class="round-strip">'+restartButtons+'</div>'+ 
    '<h3>Chọn vòng Lloyd</h3><div class="round-strip">'+jumpButtons+'</div>'+ 
    '<div class="player"><button id="km-first" '+(kmStep===0?'disabled':'')+'>|← Khởi tạo</button><button id="km-prev" '+(kmStep===0?'disabled':'')+'>← Vòng trước</button><span>'+esc(frame.label)+' · bước '+(kmStep+1)+'/'+h.length+'</span><button id="km-next" '+(kmStep===h.length-1?'disabled':'')+'>Vòng tiếp →</button><button id="km-last" '+(kmStep===h.length-1?'disabled':'')+'>Đến cuối →|</button></div>'+plot(frame)+
    '<div class="note">'+esc(frame.note)+' '+esc(traceNote)+'</div>'+summary+
    '<div class="subhead"><h3>1 · Khoảng cách mỗi câu tới TẤT CẢ tâm</h3><span class="small">Ô cam đánh dấu tâm mà K-means thực tế gán cho câu ở vòng này.</span></div>'+table(['Câu','Văn bản',...centerHeaders,'Cụm được gán','d² tới tâm được gán'],rows)+
    (frame.has_full_distances===false?'<div class="note warn">Ma trận khoảng cách tất cả tâm bị lược bớt cho lần khởi tạo này để giới hạn kích thước phản hồi. Lần thắng có đầy đủ ma trận.</div>':'')+
    updateContent+
    '<div class="subhead"><h3>2 · Nhật ký tất cả vòng Lloyd</h3><span class="small">Dòng xanh nhạt là vòng đang xem.</span></div>'+table(['Vòng','Tổng dịch tâm²','Inertia sau vòng','Điều kiện dừng','Trạng thái'],historyRows);
}

function centroid(){
  const reps=DATA.representatives,cent=DATA.centroid,rawSteps=cent.process||[];
  const cosineMode=cent.selection_mode==='cosine';
  // The UI runs in cosine mode: keep every jump button, detail panel, and ranking table in that same order.
  const steps=rawSteps.slice().sort((a,b)=>cosineMode
    ? (Number(b.cosine_score)-Number(a.cosine_score))||(Number(a.index)-Number(b.index))
    : (Number(a.rank)-Number(b.rank))||(Number(a.index)-Number(b.index)));
  const orderedRanking=(cent.ranking||[]).slice().sort((a,b)=>cosineMode
    ? (Number(b.cosine_score)-Number(a.cosine_score))||(Number(a.index)-Number(b.index))
    : (Number(b.score)-Number(a.score))||(Number(a.index)-Number(b.index)));
  const repRows=reps.map(r=>'<tr><td><b>'+esc(r.id)+'</b></td><td class="sentence">'+esc(r.text)+'</td><td>Cụm '+(r.cluster+1)+'</td><td>'+r.members.map(esc).join(', ')+'</td></tr>');
  const current=steps[Math.min(centroidStep,Math.max(steps.length-1,0))];
  const rankLabel=cosineMode?'Cosine centroid':'Điểm CentroidBow';
  const overviewRows=orderedRanking.map(x=>{
    const score=cosineMode?x.cosine_score:x.score;
    const activeRecord=steps.find(step=>Number(step.index)===Number(x.index));
    const redundancy=activeRecord?.redundancy;
    const jumpIndex=steps.findIndex(step=>Number(step.index)===Number(x.index));
    return '<tr class="'+(x.selected?'won':'')+'"><td><b>'+esc(x.id)+'</b></td><td class="sentence">'+esc(x.text)+'</td><td class="num">'+num(score,7)+'</td><td class="num">'+num(redundancy,7)+'</td><td>'+(x.selected?'<span class="pill">Được giữ</span>':'Loại')+'</td><td>'+esc(x.decision)+'</td><td><button type="button" data-centroid-jump="'+jumpIndex+'">Xem phép tính</button></td></tr>';
  });
  let html=cardTitle('03','Chọn câu đại diện và CentroidBow · xem từng thành phần',
    'Lấy câu đại diện từ các cụm K-means → xây dựng vector chủ đề μ → tính cosine cho mọi ứng viên → xếp hạng giảm dần → lần lượt áp dụng ngưỡng cosine và bộ lọc trùng lặp.')+
    metrics([[reps.length,'câu đại diện từ K-means'],[cent.selected.length,'câu được giữ để chuyển sang MMR'],[steps.length,'ứng viên được chấm'],[DATA.mmr.pool.length,'câu vào MMR']])+ 
    '<h3>Câu đại diện lấy từ mỗi cụm</h3>'+table(['ID câu gốc','Văn bản','Cụm','Các câu trong cụm'],repRows);
  if(!current)return html+'<div class="note warn">Không có nhật ký chi tiết CentroidBow trả về.</div>';
  const jumpButtons=steps.map((x,i)=>'<button type="button" data-centroid-jump="'+i+'" class="'+(i===centroidStep?'current':(i<centroidStep?'done':''))+'">'+(i+1)+'. '+esc(x.id)+'</button>').join('');
  const termRows=current.terms.map(t=>'<tr><td><b>'+esc(t.term)+'</b></td><td class="num">'+t.tf+'</td><td class="num">'+t.df+'</td><td class="num">'+num(t.idf,7)+'</td><td class="num">'+num(t.tfidf,7)+'</td><td class="num">'+num(t.centroid_sum,7)+'</td><td class="num">'+num(t.centroid_normalized,7)+'</td><td class="num">'+num(t.centroid_after_threshold,7)+'</td></tr>');
  const comparisonRows=(current.comparisons||[]).map(c=>'<tr class="'+(Number(c.similarity)>Number(c.threshold??cent.threshold)?'rejected':'')+'"><td>'+esc(c.against_id)+'</td><td class="sentence">'+esc(c.against_text)+'</td><td class="num">'+num(c.similarity,7)+'</td><td class="num">'+num(c.threshold??cent.threshold,4)+'</td><td>'+(Number(c.similarity)>Number(c.threshold??cent.threshold)?'<span class="rejected">Vượt ngưỡng → loại</span>':'<span class="passed">Không vượt ngưỡng</span>')+'</td></tr>');
  const actualNote='<div class="note">'+esc(current.actual_decision_note||'')+'</div>';
  const finalDecision=current.selected
    ? '<div class="winner-box"><b>✓ Kết quả: GIỮ '+esc(current.id)+'</b><br>'+(cosineMode?'Câu đạt ngưỡng cosine và không trùng quá mức với các câu đã giữ; câu được chuyển sang MMR.':'Câu nằm trong danh sách do CentroidBow gốc chọn.')+'<br><span class="small">Giải thích: '+esc(current.reason||'')+'</span></div>'
    : '<div class="note warn"><b>Kết quả: LOẠI '+esc(current.id)+'</b><br>'+esc(current.reason||'Ứng viên không nằm trong kết quả được chọn.')+'</div>';
  const fullCentroidRows=(cent.all_centroid_terms||[]).map(t=>'<tr><td>'+esc(t.term)+'</td><td class="num">'+num(t.raw_sum,7)+'</td><td class="num">'+num(t.normalized,7)+'</td><td class="num">'+num(t.after_threshold,7)+'</td><td>'+(Number(t.after_threshold)===0?'<span class="muted">Bị ngưỡng loại</span>':'Được giữ trong centroid')+'</td></tr>');
  const thisRank=centroidStep+1;
  html+='<div class="subhead"><h3>'+(cosineMode?'Từng ứng viên theo thứ hạng cosine':'Từng ứng viên theo thứ hạng CentroidBow')+'</h3><span class="small">Hạng '+thisRank+' / '+steps.length+'</span></div><div class="round-strip">'+jumpButtons+'</div>'+ 
    '<div class="player"><button id="centroid-first" '+(centroidStep===0?'disabled':'')+'>|← Đầu</button><button id="centroid-prev" '+(centroidStep===0?'disabled':'')+'>← Ứng viên trước</button><span>'+esc(current.id)+' · hạng '+thisRank+'</span><button id="centroid-next" '+(centroidStep===steps.length-1?'disabled':'')+'>Ứng viên tiếp →</button><button id="centroid-last" '+(centroidStep===steps.length-1?'disabled':'')+'>Cuối →|</button></div>'+ 
    '<div class="query"><b>ỨNG VIÊN ĐANG XÉT</b><b style="font-size:15px;letter-spacing:0;color:var(--ink)">'+esc(current.id)+' · '+esc(current.text)+'</b><div class="small">Hạng '+thisRank+' / '+steps.length+' · '+(cosineMode?'cos(<i>x</i><sub>i</sub>, μ)':'CentroidBow.similarity()')+' = <b>'+num(cosineMode?current.cosine_score:current.score,8)+'</b></div></div>'+ 
    '<div class="component-grid"><section class="component-card"><h4>1 · TF: tần suất token</h4><p>Đếm số lần token <i>t</i> xuất hiện trong câu <i>c</i>.</p><div class="formula math-formula"><span class="eq-line"><span class="math-op">TF</span>(<i>t</i>, <i>c</i>) = số lần <i>t</i> xuất hiện trong <i>c</i></span></div></section><section class="component-card"><h4>2 · DF và IDF</h4><p><i>DF(t)</i> là số câu đại diện có chứa token <i>t</i>; <i>N</i> là số câu đại diện được đưa vào tính TF-IDF.</p><div class="formula math-formula"><span class="eq-line"><span class="math-op">IDF</span>(<i>t</i>) = log <span class="math-frac"><span><i>N</i></span><span><span class="math-op">DF</span>(<i>t</i>)</span></span> + 1</span></div></section><section class="component-card"><h4>3 · TF-IDF</h4><p>TF-IDF ở đây không chuẩn hóa từng dòng và không dùng sublinear TF.</p><div class="formula math-formula"><span class="eq-line"><span class="math-op">TF-IDF</span>(<i>c</i>, <i>t</i>) = <span class="math-op">TF</span>(<i>t</i>, <i>c</i>) × <span class="math-op">IDF</span>(<i>t</i>)</span></div></section><section class="component-card"><h4>4 · Vector chủ đề μ</h4><p>Cộng TF-IDF theo token trên các câu đại diện, chuẩn hóa theo trọng số lớn nhất rồi áp ngưỡng <i>θ</i><sub>topic</sub>.</p><div class="formula math-formula"><span class="eq-line"><i>s</i>(<i>t</i>) = ∑<sub><i>c</i>∈R</sub> <span class="math-op">TF-IDF</span>(<i>c</i>, <i>t</i>)</span><span class="eq-line"><i>w</i>(<i>t</i>) = <span class="math-frac"><span><i>s</i>(<i>t</i>)</span><span>max<sub><i>u</i>∈V</sub> <i>s</i>(<i>u</i>)</span></span></span><span class="eq-line">μ(<i>t</i>) = 0 nếu <i>w</i>(<i>t</i>) ≤ <i>θ</i><sub>topic</sub>; ngược lại μ(<i>t</i>) = <i>w</i>(<i>t</i>)</span></div></section></div>'+ 
    '<h3>Chi tiết từng token của ứng viên '+esc(current.id)+'</h3>'+table(['Token','TF','DF','IDF','TF-IDF câu','Tổng TF-IDF','Sau chuẩn hóa','Trọng số μ(t)'],termRows)+
    '<div class="flow-focus good"><b>Cosine của '+esc(current.id)+'</b><span class="math-equation"><span class="eq-line">cos(<i>x</i><sub>i</sub>, μ) = <span class="math-frac"><span><i>x</i><sub>i</sub> · μ</span><span>‖<i>x</i><sub>i</sub>‖<sub>2</sub> ‖μ‖<sub>2</sub></span></span> = '+num(current.cosine_score,9)+'</span></span>Điểm dùng để xếp hạng là cosine trong chế độ hiện tại; sau khi tính đủ mọi câu mới áp dụng ngưỡng và lọc trùng.</div>'+ 
    '<details class="details"><summary>Toàn bộ trọng số của vector chủ đề μ ('+(cent.all_centroid_terms||[]).length+' token)</summary>'+table(['Token','Tổng s(t)','Trọng số w(t)','Sau ngưỡng μ(t)','Trạng thái'],fullCentroidRows)+'</details>'+ 
    '<details class="details"><summary>Mã nguồn similarity() của CentroidBow gốc</summary><pre class="formula" style="white-space:pre-wrap">'+esc(cent.similarity_source||'Không thể đọc source bằng inspect.getsource.')+'</pre><p class="small">Hàm này được dùng cho bộ lọc trùng lặp cặp câu; không phải công thức cosine với vector chủ đề μ.</p></details>'+ 
    '<h3>Lọc trùng với những câu đã giữ trước đó</h3><p class="help">Chỉ ứng viên đạt ngưỡng cosine mới đến bước này. So ứng viên với từng câu đã giữ; nếu <span class="math-equation"><span class="math-op">sim</span>(<i>c</i><sub>i</sub>, <i>s</i><sub>j</sub>) &gt; τ</span> thì loại ứng viên.</p>'+ 
    (current.comparisons&&current.comparisons.length?table(['Câu đã giữ','Nội dung câu đã giữ','sim(cᵢ, sⱼ)','Ngưỡng τ','Kết quả so sánh'],comparisonRows):'<div class="note">'+esc(current.reason||'Không phát sinh phép so trùng lặp cho ứng viên này.')+'</div>')+
    finalDecision+actualNote+
    '<h3>Tổng kết tất cả ứng viên</h3>'+table(['ID','Văn bản',rankLabel,'Độ trùng cao nhất','Quyết định','Lý do','Chi tiết'],overviewRows)+
    (DATA.mmr.pool.length?'<div class="query"><b>CÂU ĐẦU VÀO MMR</b>'+esc(DATA.mmr.pool[0].id)+' · '+esc(DATA.mmr.pool[0].text)+'</div>':'<div class="note warn">Không có câu sau Centroid để khởi tạo Q.</div>');
  return html;
}

function mmr(){
  const pool=DATA.mmr.pool,rounds=DATA.mmr.rounds||[],round=rounds[mmrStep];
  const lambda=Number(DATA.mmr.lambda);
  let html=cardTitle('04','MMR · xem toàn bộ các lượt tính và chọn','Theo dõi đúng 5 bước: khởi tạo R; chấm Rel và Red cho mọi ứng viên; tính điểm MMR; chọn điểm cao nhất; cập nhật R và kiểm tra dừng. Q giữ nguyên qua các vòng, còn Red phải tính lại mỗi khi R thay đổi.')+
    metrics([[pool.length,'câu sau CentroidBow'],[DATA.mmr.selected.length,'câu kết quả MMR'],[rounds.length,'lượt chọn thực tế'],[DATA.mmr.target,'target theo chế độ']]);
  if(!pool.length)return html+'<div class="note warn">Không có ứng viên sau CentroidBow nên không thể khởi tạo truy vấn Q cho MMR.</div>';
  html+='<div class="query"><b>QUERY Q · CỐ ĐỊNH QUA CÁC LƯỢT</b><b style="font-size:15px;letter-spacing:0;color:var(--ink)">'+esc(pool[0].id)+' · '+esc(pool[0].text)+'</b><div class="small">Q là câu đầu tiên sau CentroidBow. Theo cách khởi tạo của pipeline này, R₀={Q}; các câu khác là ứng viên U. Sau mỗi lượt, câu thắng được thêm vào R và bỏ khỏi U. Q không đổi; IDF lấy từ MMR.IDFs() của repository.</div></div>'+ 
    '<div class="formula">Rel(c) = sentenceSim(c, Q)<br>Red(c,R) = max<sub>r ∈ R</sub> sentenceSim(c, r)<br>MMR(c) = λ × Rel(c) − (1−λ) × Red(c,R)<br><span class="small">λ = '+num(lambda,4)+'</span></div>';
  if(!round){
    html+='<div class="note warn">MMR giữ ceil(0,7 × C) câu, với C là số ứng viên sau Centroid. Mục tiêu đã bao gồm câu mốc Q; khi mục tiêu bằng 1 thì không có lượt chọn thêm.</div><div class="winner-box"><b>Tập R đã chọn cuối cùng</b>'+chips(DATA.mmr.selected.map(x=>x.id+' · '+x.text))+'</div>';
    return html;
  }
  const jumpButtons=rounds.map((r,i)=>'<button type="button" data-mmr-jump="'+i+'" class="'+(i===mmrStep?'current':(i<mmrStep?'done':''))+'">Vòng '+(i+1)+'</button>').join('');
  html+='<div class="subhead"><h3>Điều hướng từng vòng MMR</h3><span class="small">Vòng '+(mmrStep+1)+' / '+rounds.length+'</span></div><div class="round-strip">'+jumpButtons+'</div>'+ 
    '<div class="player"><button id="mmr-first" '+(mmrStep===0?'disabled':'')+'>|← Đầu</button><button id="mmr-prev" '+(mmrStep===0?'disabled':'')+'>← Vòng trước</button><span>Vòng '+(mmrStep+1)+'/'+rounds.length+'</span><button id="mmr-next" '+(mmrStep===rounds.length-1?'disabled':'')+'>Vòng tiếp →</button><button id="mmr-last" '+(mmrStep===rounds.length-1?'disabled':'')+'>Cuối →|</button></div>'+ 
    '<div class="component-grid"><section class="component-card"><h4>Trước lượt này: R đã chọn</h4>'+chips(round.selected_before.map(x=>x.id+' · '+x.text))+'<p>Mỗi ứng viên được so với từng câu trong tập này khi tính Red.</p></section><section class="component-card"><h4>Sau lượt này: R đã chọn</h4>'+chips((round.selected_after||[]).map(x=>x.id+' · '+x.text))+'<p>Câu thắng là <b>'+esc(round.winner_id)+'</b>. Tập R này chuyển sang đầu vào của lượt sau.</p></section><section class="component-card"><h4>Ứng viên còn lại sau lượt</h4>'+chips((round.remaining_after||[]).map(x=>x.id+' · '+x.text))+'<p>Câu thắng bị loại khỏi danh sách ứng viên và không bị chấm lại.</p></section></div>';
  const rows=round.candidates.map(c=>'<tr class="'+(c.winner?'won':'')+'"><td><b>'+esc(c.id)+'</b><br><span class="small">'+esc(c.text)+'</span></td><td class="num">'+num(c.rel,7)+'</td><td class="num">'+num(c.red,7)+'<br><span class="small">max theo R</span></td><td class="num"><b>'+num(c.formula_score,7)+'</b></td><td class="num"><b>'+num(c.score,7)+'</b></td><td>'+(c.winner?'<span class="pill">★ Thắng lượt</span>':'Không thắng lượt này')+'</td></tr>');
  html+=mmrRoundNetwork(round)+flowCaption('Điểm MMR của mọi ứng viên vòng '+(mmrStep+1),'Thanh dài hơn biểu thị điểm cao hơn trong chính vòng này')+mmrScoreBars(round.candidates)+'<div class="subhead"><h3>Chấm TẤT CẢ ứng viên còn lại ở vòng '+(mmrStep+1)+'</h3><span class="small">Ô màu cam trong phần chi tiết = Sim(c,r) lớn nhất được dùng làm Red.</span></div>'+table(['Ứng viên c','Rel = Sim(c,Q)','Red = max Sim(c,r)','MMR theo công thức','MMRScore() thật','Quyết định'],rows)+ 
    '<h3>Chi tiết phép tính của từng ứng viên</h3><p class="help">Mở từng ứng viên để xem tử số/mẫu số của sentenceSim, mọi phép so sánh Red với các câu trong R, phép lấy max và phép tính MMR.</p>'+round.candidates.map(candidateWalkthrough).join('')+
    '<div class="winner-box"><b>✓ Kết thúc vòng '+(mmrStep+1)+': chọn '+esc(round.winner_id)+'.</b><br>Ứng viên này đã được thêm vào R. '+(mmrStep<rounds.length-1?'Lượt sau dùng R mới ở trên và tính lại Red cho toàn bộ ứng viên còn lại.':'Đây là lượt chọn cuối trong lần chạy này.')+'</div>';
  const allRoundRows=rounds.map((r,i)=>'<tr class="'+(i===mmrStep?'won':'')+'"><td>Vòng '+(i+1)+'</td><td>'+r.selected_before.map(x=>esc(x.id)).join(', ')+'</td><td>'+r.candidates.map(c=>esc(c.id)+' = '+num(c.score,5)+(c.winner?' ★':'')).join('<br>')+'</td><td><b>'+esc(r.winner_id)+'</b></td><td>'+(r.selected_after||[]).map(x=>esc(x.id)).join(', ')+'</td><td>'+(r.remaining_after||[]).map(x=>esc(x.id)).join(', ')||'—'+'</td><td><button type="button" data-mmr-jump="'+i+'">Mở vòng</button></td></tr>');
  html+='<details class="details"><summary>Nhật ký tổng kết TẤT CẢ '+rounds.length+' vòng MMR</summary><p class="small">Mỗi dòng có tập R đầu vào, điểm của mọi ứng viên, câu thắng, R sau lượt và các ứng viên còn lại. Bấm “Mở vòng” để đến đúng các phép tính.</p>'+table(['Lượt','R trước','Điểm các ứng viên (★ thắng)','Câu thắng','R sau','Còn lại','Điều hướng'],allRoundRows)+'</details>'+
    '<h3>Kết quả cuối cùng của MMR</h3><div class="winner-box"><b>Thứ tự câu MMR đã chọn</b><ol class="final-list">'+DATA.mmr.selected.map(x=>'<li><small>'+esc(x.id)+'</small>'+esc(x.text)+'</li>').join('')+'</ol></div>';
  return html;
}

function finalStage(){const rows=DATA.final.map((s,i)=>'<li><small>'+esc(s.id)+' · tài liệu '+esc(s.doc_label)+'</small>'+esc(s.text)+'</li>');return cardTitle('05','Phục hồi vị trí và bản tóm tắt','Giữ các câu được MMR chọn theo tỷ lệ 70%, áp dụng giới hạn số câu nếu có rồi sắp theo thứ tự nguồn; không bổ sung lại câu đã loại.')+metrics([[DATA.mmr.selected.length,'MMR chọn'],[DATA.final.length,'câu cuối'],[DATA.final.length===DATA.final_target?'đạt':'đạt một phần','len_summary='+DATA.final_target]])+'<div class="note">MMR order: '+chips(DATA.mmr.selected.map(x=>x.id))+'</div><ol class="final-list">'+rows.join('')+'</ol>'+(DATA.final.length<DATA.final_target?'<div class="note warn">Pipeline chỉ có '+DATA.final.length+' câu đại diện hợp lệ, ít hơn len_summary mục tiêu '+DATA.final_target+'.</div>':'');}

function flowAdd(group,type,title,description,extra={}){flowSteps.push({group,type,title,description,...extra});}
function flowBuild(){
  flowSteps=[];
  flowAdd(0,'input','Văn bản đầu vào','Đọc các tài liệu gốc trước khi chúng được tách câu và chuẩn hóa.');
  flowAdd(0,'preprocess','Tách câu và tiền xử lý','Tách câu, chuyển chữ thường, tách từ và loại câu có không quá 4 token theo pipeline.');
  flowAdd(0,'wordvec','Biểu diễn câu bằng Word2Vec','Mỗi câu được đổi thành vector Word2Vec 256 chiều; PCA 2D chỉ được dùng để hiển thị trên biểu đồ.');

  // K-means begins with the complete point cloud, then places distinct K-means++ centers,
  // then animates point-by-point assignment and the old-to-new center movement each round.
  const kmRuns=DATA.kmeans.restarts||[];
  const kmBestIndex=Math.max(0,kmRuns.findIndex(r=>r.best));
  const kmBestRun=kmRuns[kmBestIndex]||{history:DATA.kmeans.history||[],run:DATA.kmeans.best_run||1,best:true};
  const kmHistory=kmBestRun.history||DATA.kmeans.history||[];
  if(kmHistory.length){
    flowAdd(1,'km-points','K-Means · hiện toàn bộ điểm','Mỗi chấm là một câu sau khi đổi thành vector. Chưa có cụm và chưa có màu phân loại.');
    flowAdd(1,'km-init','K-Means · chọn K tâm ban đầu','K tâm K-means++ xuất hiện dưới dạng hình thoi, khác hình tròn biểu diễn các câu.');
    // The initial state (iteration 0) is already shown in the separate K-means++
    // initialization scene above, so list only real Lloyd iterations here.
    kmHistory.slice(1).forEach((frame,i)=>{
      const meta={runIndex:kmBestIndex,frameIndex:i+1,kmFrameIndex:i+1};
      flowAdd(1,'km-frame','K-Means · vòng '+(frame.iteration??(i+1))+': gán từng câu',
        'Duyệt từng câu, so khoảng cách tới các tâm và gán câu vào tâm gần nhất theo vector 256D.',meta);
      flowAdd(1,'km-move','K-Means · vòng '+(frame.iteration??(i+1))+': dịch chuyển tâm',
        'Xem riêng từng tâm dịch từ vị trí cũ đến trung bình vector mới của các câu trong cụm.',meta);
    });
    flowAdd(1,'km-final','K-Means · kết quả cuối và đường nối tâm–điểm',
      'Hiển thị các cụm cuối cùng, với đường nối mỗi câu tới đúng tâm cụm mà nó được gán vào.');
  }else flowAdd(1,'km-animation','K-Means · phân cụm qua từng vòng','Dùng điều khiển biểu đồ để xem khởi tạo tâm, gán câu, cập nhật tâm và điều kiện hội tụ.');

  // Show only the representatives chosen by the real pipeline; do not step through
  // every sentence-to-centroid comparison in a separate scene.
  flowAdd(2,'representative-summary','Các câu đại diện','Tổng hợp câu đại diện được pipeline chọn cho từng cụm.');

  // Centroid lesson follows the algorithmic order: input -> centroid vector -> all cosine calculations
  // -> rank by cosine -> walk the ranked list and apply the cosine/redundancy thresholds.
  const centroidItems=DATA.centroid.process||[];
  flowAdd(3,'centroid-overview','Bước 1/5 · Xác định tập câu đầu vào','Chỉ nhận và hiển thị các câu đại diện từ K-means; ở bước này chưa tạo vector chủ đề, chưa tính cosine, chưa xếp hạng và chưa lọc câu.');
  flowAdd(3,'centroid-vector','Bước 2/5 · Xây dựng vector chủ đề μ','Tính TF-IDF trên tập câu đại diện; cộng trọng số theo token, chuẩn hóa theo giá trị lớn nhất và đặt các trọng số dưới ngưỡng về 0.');
  flowAdd(3,'centroid-candidate','Bước 3/5 · Tính cosine cho mọi câu','Giữ nguyên thứ tự đầu vào để tính tích vô hướng, chuẩn của từng vector và cosine; bảng tính chi tiết nằm cuối bước này.');
  flowAdd(3,'centroid-ranking','Bước 4/5 · Xếp hạng theo cosine','Sắp xếp toàn bộ ứng viên theo cosine giảm dần. Đây chỉ là thứ tự ưu tiên, chưa phải danh sách giữ cuối cùng và chưa lọc trùng.');
  (DATA.centroid.ranking||[]).slice().sort((a,b)=>(Number(b.cosine_score)-Number(a.cosine_score))||(Number(a.index)-Number(b.index))).forEach((rankRow,rankIndex)=>{
    const processIndex=centroidItems.findIndex(x=>Number(x.index)===Number(rankRow.index));
    if(processIndex>=0) flowAdd(3,'centroid-decision','Bước 5 · Duyệt hạng '+(rankIndex+1)+'/'+(DATA.centroid.ranking||[]).length+': '+rankRow.id,'Duyệt theo thứ hạng cosine: cosine dưới θ thì loại; đạt θ thì kiểm tra trùng với các câu đã giữ bằng sim(cᵢ, sⱼ) và ngưỡng τ.',{centroidIndex:processIndex,centroidRank:rankIndex});
  });

  if(!DATA.mmr.pool.length){
    flowAdd(4,'mmr-empty','MMR · Không có câu sau Centroid','Centroid không trả về ứng viên nên không thể khởi tạo Q.');
  }else{
    flowAdd(4,'mmr-query','MMR · Khởi tạo R','Chọn Q cố định và khởi tạo R = {Q}; các câu còn lại nằm trong tập ứng viên U.',{roundIndex:-1});
    const mmrRounds=DATA.mmr.rounds||[];
    if(!mmrRounds.length){
      flowAdd(4,'mmr-loop-check','MMR · Kiểm tra điều kiện dừng','Không có lượt chọn bổ sung trong cấu hình hiện tại.',{roundIndex:-1});
    }
    mmrRounds.forEach((r,ri)=>{
      (r.candidates||[]).forEach((c,ci)=>{
        flowAdd(4,'mmr-candidate-rel','Vòng '+(ri+1)+' · Rel('+c.id+')','Đang tính độ liên quan của câu '+c.id+' với Q cố định.',{roundIndex:ri,candidateIndex:ci});
        flowAdd(4,'mmr-candidate-red','Vòng '+(ri+1)+' · Red('+c.id+')','Đang so câu '+c.id+' với từng câu đang có trong R và lấy similarity lớn nhất.',{roundIndex:ri,candidateIndex:ci});
        flowAdd(4,'mmr-candidate-score','Vòng '+(ri+1)+' · MMR('+c.id+')','Đang thay Rel và Red vào công thức MMR cho câu '+c.id+'.',{roundIndex:ri,candidateIndex:ci});
      });
      flowAdd(4,'mmr-winner','Vòng '+(ri+1)+' · Chọn câu thắng','Sau khi chấm hết ứng viên, chọn câu có MMRScore thực tế cao nhất và chuyển câu đó sang R.',{roundIndex:ri});
      flowAdd(4,'mmr-loop-check','Vòng '+(ri+1)+' · Cập nhật R / kiểm tra dừng','Cập nhật R và U. Nếu chưa dừng, vòng kế tiếp sẽ chấm lại ứng viên theo R mới.',{roundIndex:ri});
    });
  }
  flowAdd(5,'mmr-order','Thứ tự ưu tiên do MMR chọn','Đây là thứ tự câu thắng theo từng lượt MMR, trước khi phục hồi vị trí nguồn.');
  flowAdd(5,'final','Phục hồi vị trí · bản tóm tắt cuối','Dùng add_position của pipeline để đưa các câu về thứ tự tài liệu gốc và tạo bản tóm tắt.');
}
function rowTable(headers,rows){return table(headers,rows);}
function flowCaption(title,sub=''){return '<div class="flow-table-caption"><h3>'+title+'</h3><span>'+sub+'</span></div>';}
function flowBadge(text){return '<div class="flow-substep"><b>ĐANG THỰC HIỆN</b> '+esc(text)+'</div>';}

function mmrRoadStepFor(item){
  if(item.type==='mmr-query'||item.type==='mmr-empty')return 1;
  if(item.type==='mmr-rel-overview'||item.type==='mmr-red-overview')return 2;
  if(item.type==='mmr-score-overview'||item.type==='mmr-candidate-evaluate')return 3;
  if(item.type==='mmr-winner')return 4;
  if(item.type==='mmr-loop-check')return 5;
  return 1;
}
function mmrRoadmap(activeStep,roundIndex){
  const steps=[
    ['Khởi tạo R','Chọn câu đầu tiên'],
    ['Đánh giá ứng viên','Rel và Red'],
    ['Tính điểm MMR','Thưởng − phạt'],
    ['Chọn câu cao nhất','argmax điểm'],
    ['Cập nhật & dừng','Lặp lại nếu cần']
  ];
  const noRounds=roundIndex===-1;
  const label=(Number.isInteger(roundIndex)&&roundIndex>=0)?'VÒNG '+(roundIndex+1):'KHỞI TẠO / ĐIỀU KIỆN DỪNG';
  const tiles=steps.map((step,i)=>{
    const n=i+1,cls=(n===activeStep?' active':((!noRounds&&n<activeStep)||(noRounds&&n===1)?' done':''));
    return '<div class="mmr-road-step'+cls+'"><span class="mmr-road-no">'+String(n).padStart(2,'0')+'</span><span><b>'+step[0]+'</b><small>'+step[1]+'</small></span></div>';
  }).join('');
  return '<section class="mmr-roadmap" aria-label="Trình tự 5 bước MMR"><div class="mmr-roadmap-head"><div><strong>MMR · Trình tự 5 bước trong mỗi vòng</strong><small>Đọc từ trái sang phải; ở bước 5, nếu chưa dừng thì quay lại bước 2.</small></div><span class="mmr-roadmap-round">'+label+'</span></div><div class="mmr-road-steps">'+tiles+'</div><div class="mmr-road-loop-note"><b>Điểm mấu chốt của vòng lặp:</b> Q giữ nguyên; R được bổ sung câu thắng. Vì vậy, Rel so với Q thường giữ nguyên giá trị, còn Red phải tính lại theo R mới nên điểm MMR của các câu còn lại có thể thay đổi.</div></section>';
}

function graphFlow(items){return '<div class="algorithm-graph">'+items.map((n,i)=>(i?'<div class="algorithm-arrow" aria-hidden="true">→</div>':'')+'<div class="algorithm-node '+esc(n.kind||'')+'"><span class="node-no">'+String(i+1).padStart(2,'0')+'</span><b>'+esc(n.title)+'</b><small>'+esc(n.body||'')+'</small></div>').join('')+'</div>';}
function cosineInputThreshold(){const n=Number($('cosine-threshold')?.value??0.35);return Number.isFinite(n)?Math.max(0,Math.min(1,n)):0.35;}
function cosineThreshold(){const n=Number(DATA?.centroid?.cosine_threshold??0.35);return Number.isFinite(n)?Math.max(0,Math.min(1,n)):0.35;}
function cosineDecision(score){const threshold=cosineThreshold(),value=Number(score),valid=Number.isFinite(value),pass=valid&&value>=threshold;return {threshold,value,valid,pass,label:pass?'Đạt ngưỡng cosine':'Dưới ngưỡng cosine'};}
function mmrScoreBars(candidates){const finite=candidates.map(c=>Number(c.score)).filter(Number.isFinite),lo=finite.length?Math.min(...finite):0,hi=finite.length?Math.max(...finite):1,range=hi-lo||1;return '<div class="mmr-bars">'+candidates.slice().sort((a,b)=>Number(b.score)-Number(a.score)).map(c=>{const v=Number(c.score),valid=Number.isFinite(v),width=valid?Math.max(4,18+82*(v-lo)/range):3;return '<div class="mmr-bar-row '+(c.winner?'winner':'')+'"><div class="mmr-bar-label">'+esc(c.id)+(c.winner?' ★':'')+'</div><div class="mmr-bar-track"><div class="mmr-bar-fill" style="width:'+width+'%"></div></div><div class="mmr-bar-value">'+num(c.score,7)+'</div></div>';}).join('')+'</div>';}
function mmrRoundNetwork(round){const q=DATA.mmr.pool[0],before=round.selected_before||[],after=round.selected_after||[],remaining=round.remaining_after||[];return '<div class="mmr-network"><div class="mmr-network-grid"><div class="mmr-network-node qnode"><b>Q · truy vấn cố định</b><p>'+esc(q?.id||'Q')+' — '+esc(q?.text||'')+'</p></div><div class="mmr-network-node snode"><b>R trước lượt</b><p>'+(before.map(x=>esc(x.id)).join(' · ')||'∅')+'</p><p class="small">Các câu dùng để tính Red.</p></div><div class="mmr-network-node cnode"><b>Ứng viên U trước lượt</b><p>'+(round.remaining_before||[]).map(x=>esc(x.id)).join(' · ')+'</p><p class="small">Tính Rel và Red lại cho từng câu.</p></div></div><div class="mmr-network-arrow">↓</div><div class="mmr-network-grid"><div class="mmr-network-node"><b>Rel(c)</b><p>So ứng viên với Q.</p></div><div class="mmr-network-node"><b>Red(c,R)</b><p>Lấy max similarity với mọi câu trong R.</p></div><div class="mmr-network-node"><b>MMRScore(c)</b><p>λ × Rel − (1−λ) × Red.</p></div></div><div class="mmr-network-arrow">↓</div><div class="mmr-network-grid"><div class="mmr-network-node winnode"><b>★ Câu thắng: '+esc(round.winner_id)+'</b><p>Điểm cao nhất trong lượt.</p></div><div class="mmr-network-node snode"><b>R sau lượt</b><p>'+(after.map(x=>esc(x.id)).join(' · ')||'∅')+'</p><p class="small">R mới = R cũ ∪ {câu thắng}.</p></div><div class="mmr-network-node cnode"><b>U còn lại</b><p>'+(remaining.map(x=>esc(x.id)).join(' · ')||'Không còn câu')+'</p><p class="small">Câu thắng bị loại khỏi ứng viên.</p></div></div></div>';}

function flowInput(){const sourceDocs=RUN_DOCS.slice();return metrics([[sourceDocs.length,'tài liệu đầu vào'],[DATA.sentences.length,'câu hợp lệ sau tiền xử lý'],[DATA.dropped_count,'câu bị loại']])+'<div class="flow-focus"><b>Văn bản trước xử lý</b>Các tài liệu này là nguồn câu để pipeline tách và xử lý ở bước tiếp theo.</div>'+sourceDocs.map((text,i)=>'<section class="component-card"><h4>Tài liệu '+(i+1)+'</h4><p style="white-space:pre-wrap">'+esc(text||'(tài liệu trống)')+'</p></section>').join('');}
function flowPreprocess(){const s=DATA.sentences;return metrics([[s.length,'câu hợp lệ'],[DATA.documents,'tài liệu'],[DATA.dropped_count,'câu bị loại']])+sentenceTable(s)+(DATA.boundaries.length?'<div class="note">Chỉ số bắt đầu tài liệu (last_indexs): '+DATA.boundaries.map(x=>esc(x)).join(', ')+'</div>':'')+'<div class="flow-equation">Quy tắc chính: tách câu → chuyển thường → tách từ → loại dấu câu và token bị bỏ → bỏ câu có không quá 4 token.</div>'; }
function flowWordVec(){
  const rows=DATA.sentences.map(s=>'<tr><td><b>'+esc(s.id)+'</b></td><td class="sentence">'+esc(s.text)+'</td><td>'+chips(s.tokens)+'</td><td class="num">'+num(s.vector_norm,5)+'</td><td class="flow-vector">['+(s.vector_preview||[]).map(v=>num(v,4)).join(', ')+(s.vector_preview?.length===8?', …':'')+']</td></tr>');
  return '<div class="flow-focus"><b>Thao tác</b>Với mỗi câu, cộng vector Word2Vec của các token đã tiền xử lý: xᵢ = Σ v(token). K-means dùng toàn bộ vector gốc 256 chiều; phần preview dưới đây chỉ hiện 8 chiều đầu để đọc dễ hơn.</div><div class="flow-equation">xᵢ = v(w₁) + v(w₂) + … + v(wₘ) ∈ ℝ²⁵⁶</div>'+rowTable(['Câu','Văn bản','Token','||xᵢ||₂','8 chiều đầu của vector câu'],rows)+'<details class="flow-hidden-detail"><summary>Xem vector đầy đủ 256 chiều của từng câu</summary>'+DATA.sentences.map(s=>'<div class="details"><b>'+esc(s.id)+'</b><div class="flow-vector">['+(s.vector||[]).map(v=>num(v,6)).join(', ')+']</div></div>').join('')+'</details>';
}
function runFor(item){return (DATA.kmeans.restarts||[])[item.runIndex]||{history:DATA.kmeans.history||[],seed_indices:[],run:1,best:true,final_inertia:null,has_full_vectors:true,has_full_distances:true};}
function seedPlot(item){
  const run=runFor(item),seeds=run.seed_indices||[],shown=seeds.slice(0,item.seedPos+1),pts=DATA.sentences;
  const xs=pts.map(p=>p.x),ys=pts.map(p=>p.y),project=pcaProjector(xs,ys,42,38,610,285),X=project.x,Y=project.y;
  let out='<svg class="plot seed-plot" viewBox="0 0 700 390" role="img" aria-label="Các tâm K-means++ được chọn lần lượt"><text x="28" y="24" font-size="12" fill="#63708a">Khởi tạo tâm · đang hiển thị '+shown.length+' / '+seeds.length+' tâm đã chọn</text>';
  out+='<defs><radialGradient id="seedHalo"><stop offset="0" stop-color="#E7C58E" stop-opacity=".30"/><stop offset="1" stop-color="#E7C58E" stop-opacity="0"/></radialGradient></defs>';
  for(const p of pts){out+='<circle cx="'+X(p.x)+'" cy="'+Y(p.y)+'" r="4.2" fill="#C6CDD8" stroke="#fff" stroke-width="1.1"><title>'+esc(p.id+' · '+p.text)+'</title></circle><text x="'+(X(p.x)+8)+'" y="'+(Y(p.y)+4)+'" font-size="8" fill="#79839A">'+esc(p.id)+'</text>';}
  shown.forEach((idx,j)=>{const p=pts[idx];if(!p)return;const x=X(p.x),y=Y(p.y),color=KM_COLORS[j%KM_COLORS.length],current=j===item.seedPos;if(current)out+='<circle cx="'+x+'" cy="'+y+'" r="23" fill="url(#seedHalo)" class="point-halo"></circle>';out+='<path class="seed-marker" d="'+starPath(x,y,8.5,3.8)+'" fill="'+color+'" stroke="'+(current?'#E6C98E':'#fff')+'" stroke-width="'+(current?'2.1':'1.2')+'"><title>Tâm '+(j+1)+' = '+esc(p.id)+(current?' · tâm vừa được chọn':' · tâm đã có')+'</title></path><text x="'+(x+10)+'" y="'+(y-8)+'" font-size="10" font-weight="800" fill="#68728A">'+(j+1)+'</text>';});
  out+='</svg><div class="legend"><span>● câu chưa chọn làm tâm</span><span>★ tâm đã chọn</span><span>★ viền vàng nhạt: tâm vừa được chọn</span><span>PCA 2D chỉ để hiển thị; chọn tâm theo vector Word2Vec 256 chiều</span></div>';
  return out;
}
function kmSeedStep(item){
  const run=runFor(item),sent=DATA.sentences[item.seedIndex],idx=item.seedPos,trace=item.seedTrace;
  const weightRows=(trace?.probabilities||[]).map(p=>{const candidate=DATA.sentences[p.index];return '<tr class="'+(p.index===item.seedIndex?'won':'')+'"><td>'+esc(candidate?.id||('C'+(p.index+1)))+'</td><td class="num">'+num(p.distance,7)+'</td><td class="num">'+num(p.distance_sq,8)+'</td><td class="num">'+num(p.probability*100,5)+'%</td><td>'+(p.index===item.seedIndex?'<span class="pill">Tâm được chọn</span>':'')+'</td></tr>';});
  const trialRows=(trace?.trials||[]).map((t,j)=>'<tr class="'+(t.chosen?'won':'')+'"><td>Lượt thử '+(j+1)+'</td><td>'+esc(DATA.sentences[t.candidate_index]?.id||('C'+(t.candidate_index+1)))+'</td><td class="num">'+num(t.sample_probability*100,5)+'%</td><td class="num">'+num(t.potential,8)+'</td><td>'+(t.chosen?'<span class="pill">Giữ lại · potential nhỏ nhất</span>':'Không giữ lượt thử này')+'</td></tr>');
  let sampling='';
  if(trace?.first){sampling='<div class="flow-equation">P(xᵢ) = 1 / N = 1 / '+DATA.sentences.length+' = '+num(100/DATA.sentences.length,5)+'%</div>'+flowCaption('Các câu có xác suất chọn tâm đầu tiên bằng nhau')+rowTable(['Câu','D tới tâm trước','D²','P chọn','Kết quả'],weightRows)+'<p class="small">Tâm đầu tiên được lấy ngẫu nhiên đồng đều; ở thời điểm này chưa có tâm trước đó nên chưa tính trọng số D².</p>';}
  else if(trace && trace.probabilities?.length && item.seedTraceMatches){sampling='<div class="flow-equation">D(xᵢ) = khoảng cách nhỏ nhất tới các tâm đã có<br>P(xᵢ) = D(xᵢ)² / Σⱼ D(xⱼ)²<br>Current potential = '+num(trace.current_potential,9)+'</div>'+flowCaption('Trọng số lấy mẫu trước khi chọn tâm μ'+(idx+1),'Bảng đầy đủ các câu ứng viên')+rowTable(['Câu','D tới tâm gần nhất','D² dùng làm trọng số','P lấy mẫu','Kết quả'],weightRows)+flowCaption('Các lượt lấy mẫu cục bộ của sklearn','Chọn ứng viên làm giảm potential nhiều nhất')+rowTable(['Lượt','Ứng viên rút được','P tại lượt rút','Potential sau khi thử','Quyết định'],trialRows)+'<div class="flow-result"><b>Sklearn giữ '+esc(sent?.id||'?')+' làm tâm μ'+(idx+1)+'</b><br>Potential giảm từ '+num(trace.current_potential,8)+' xuống '+num(trace.remaining_potential,8)+'.</div>';}
  else {sampling='<div class="note warn">Phiên bản sklearn hiện tại không khớp với bản tái dựng log seed; vì vậy chỉ hiển thị seed thực tế trả về, không hiển thị xác suất trung gian chưa được xác minh.</div>';}
  return flowBadge('Khởi tạo lần '+run.run+' · tâm μ'+(idx+1))+seedPlot(item)+graphFlow([{title:'Chọn tâm đầu tiên',body:'Lấy ngẫu nhiên đồng đều một câu làm μ₁.',kind:'primary-node'},{title:'Chọn tâm tiếp theo',body:'Dùng xác suất tỉ lệ D(x)² tới tâm gần nhất.'},{title:'Lặp đến đủ K tâm',body:'K-means++ rải các seed ra các vùng khác nhau.'},{title:'Bắt đầu gán cụm',body:'Tính khoảng cách mọi câu tới tất cả tâm.'}])+'<div class="graph-caption">Theo đúng sklearn trong repository: tâm đầu lấy đồng đều ngẫu nhiên; tâm sau dùng K-means++, không phải sinh K tâm độc lập hoàn toàn ngẫu nhiên.</div><div class="flow-focus"><b>Tâm μ'+(idx+1)+' được chọn từ '+esc(sent?.id||'?')+'</b>'+esc(sent?.text||'')+'<br><span class="small">Lần khởi tạo '+run.run+'; đây là seed thực tế của sklearn K-means++.</span></div>'+sampling+(sent?.vector?'<details class="flow-hidden-detail"><summary>Xem vector 256 chiều của tâm μ'+(idx+1)+'</summary><div class="flow-vector">['+sent.vector.map(v=>num(v,6)).join(', ')+']</div></details>':'');
}
function kmAssignStep(item){
  const run=runFor(item),frame=run.history[item.frameIndex],sent=DATA.sentences[item.sentenceIndex],si=item.sentenceIndex;
  if(!frame||!sent)return '<div class="note warn">Không có dữ liệu khoảng cách cho bước này.</div>';
  const ds=(frame.distance_matrix||[])[si]||[],assigned=frame.assignments[si];
  const dRows=Array.from({length:DATA.kmeans.k},(_,j)=>'<tr class="'+(j===assigned?'won':'')+'"><td>μ'+(j+1)+'</td><td class="num">'+(ds.length?num(ds[j],8):'—')+'</td><td>'+(j===assigned?'<span class="pill">Khoảng cách nhỏ nhất · chọn</span>':'Không được chọn')+'</td></tr>');
  const title=item.initial?'Khởi tạo':'Vòng Lloyd '+frame.iteration;
  const foot=ds.length?'So sánh tất cả d² ở bảng: argmin = μ'+(assigned+1)+'.':'Lượt khởi tạo này không lưu ma trận d² đầy đủ do giới hạn dung lượng; log chỉ còn d² được chọn = '+num(frame.distances[si],8)+'.';
  return flowBadge(title+' · gán '+sent.id)+graphFlow([{title:'Điểm câu xᵢ',body:sent.id+' · vector Word2Vec 256 chiều.'},{title:'Tính K khoảng cách',body:'d²(xᵢ, μ₁), …, d²(xᵢ, μK).'},{title:'Chọn tâm gần nhất',body:'argminⱼ d²(xᵢ, μⱼ).',kind:'primary-node'},{title:'Gán vào cụm Cⱼ',body:'Nối câu với tâm gần nhất.',kind:'good-node'},{title:'Tính lại tâm cụm',body:'μⱼ mới = trung bình vector thành viên.'},{title:'Di chuyển và kiểm tra',body:frame.converged?'Đã đạt điều kiện hội tụ.':'Chưa hội tụ: gán lại với tâm mới.',kind:frame.converged?'good-node':'warn-node'}])+plot(frame,si)+'<div class="flow-focus"><b>Câu đang xét: '+esc(sent.id)+'</b>'+esc(sent.text)+'</div><div class="flow-equation">d²(xᵢ, μⱼ) = Σ<sub>r=1…256</sub> (xᵢᵣ − μⱼᵣ)²<br>Chọn cụm cᵢ = argminⱼ d²(xᵢ, μⱼ)</div>'+'<div class="flow-table-caption"><h3>Khoảng cách từ '+esc(sent.id)+' đến từng tâm</h3><span>Giá trị thực từ vector 256 chiều</span></div>'+rowTable(['Tâm','d²(x, μ)','Kết quả'],dRows)+'<div class="flow-result"><b>Kết luận: '+esc(sent.id)+' → Cụm '+(assigned+1)+'</b><br>d² được chọn = '+num(frame.distances[si],8)+'. '+foot+'</div>'+(sent.vector?'<details class="flow-hidden-detail"><summary>Xem vector câu 256 chiều</summary><div class="flow-vector">['+sent.vector.map(v=>num(v,5)).join(', ')+']</div></details>':'');
}
function kmUpdateStep(item){
  const run=runFor(item),frame=run.history[item.frameIndex],u=frame?.cluster_updates?.[item.updateIndex];
  if(!u)return '<div class="note warn">Không có thông tin cập nhật tâm cho bước này.</div>';
  const memberRows=(u.members||[]).map(i=>'<tr><td>'+esc(DATA.sentences[i]?.id||('C'+(i+1)))+'</td><td class="sentence">'+esc(DATA.sentences[i]?.text||'')+'</td><td class="num">'+num(DATA.sentences[i]?.vector_norm,5)+'</td></tr>');
  const count=u.count||0,old=u.old_preview||[],nw=u.new_preview||[];
  const shift=Number(u.shift_sq||0);
  return flowBadge('Vòng '+frame.iteration+' · cập nhật tâm cụm '+(u.cluster+1))+graphFlow([{title:'Tập thành viên C'+(u.cluster+1),body:(u.members||[]).map(i=>DATA.sentences[i]?.id||('C'+(i+1))).join(', ')||'Cụm rỗng.'},{title:'Tính trung bình',body:'Cộng vector rồi chia cho n='+count+'.'},{title:'Tâm mới μ'+(u.cluster+1),body:'Cập nhật tọa độ theo 256 chiều.',kind:'primary-node'},{title:'Đo độ dịch tâm',body:'||μ mới − μ cũ||² = '+num(shift,7)+'.'},{title:frame.converged?'Dừng':'Lặp lại',body:frame.converged?'Đạt điều kiện hội tụ.':'Gán lại toàn bộ câu với tâm mới.',kind:frame.converged?'good-node':'warn-node'}])+'<div class="flow-equation">μⱼ mới = (x₁ + x₂ + … + xₙ) / n<br>n = '+count+' câu sử dụng khi cập nhật tâm này.</div>'+flowCaption('Các câu góp phần cập nhật tâm μ'+(u.cluster+1))+(memberRows.length?rowTable(['Câu thành viên','Văn bản','Norm vector'],memberRows):'<div class="note warn">Cụm rỗng ở vòng này. Tâm được xử lý theo cơ chế cụm rỗng của scikit-learn.</div>')+'<div class="flow-table-caption"><h3>So sánh tâm trước và sau cập nhật</h3><span>8 chiều đầu</span></div>'+rowTable(['Thành phần','Tâm cũ','Tâm mới','Chênh lệch'],Array.from({length:Math.max(old.length,nw.length)},(_,i)=>'<tr><td>dim '+(i+1)+'</td><td class="num">'+num(old[i],7)+'</td><td class="num">'+num(nw[i],7)+'</td><td class="num">'+num(Number(nw[i]||0)-Number(old[i]||0),7)+'</td></tr>'))+'<div class="flow-result"><b>Độ dịch tâm² của cụm = '+num(shift,9)+'</b><br>Tổng độ dịch tâm của mọi cụm được kiểm tra ở bước kế tiếp.</div>'+(u.old_vector&&u.new_vector?'<details class="flow-hidden-detail"><summary>Xem đầy đủ 256 chiều của tâm cũ và tâm mới</summary><div class="details"><b>Tâm cũ</b><div class="flow-vector">['+u.old_vector.map(v=>num(v,6)).join(', ')+']</div></div><div class="details"><b>Tâm mới</b><div class="flow-vector">['+u.new_vector.map(v=>num(v,6)).join(', ')+']</div></div></details>':'');
}
function kmConvergeStep(item){
  const run=runFor(item),frame=run.history[item.frameIndex],prev=run.history[item.frameIndex-1];
  if(!frame)return '';
  return flowBadge('Vòng '+frame.iteration+' · xác định có dừng hay tiếp tục')+'<div class="flow-equation">Shift = Σⱼ ||μⱼ mới − μⱼ cũ||² = '+num(frame.shift,10)+'<br>Inertia = Σᵢ minⱼ ||xᵢ − μⱼ||² = '+num(frame.inertia,10)+'</div><div class="metric-row">'+metrics([[num(frame.shift,8),'tổng độ dịch tâm²'],[num(frame.inertia,8),'inertia sau cập nhật'],[frame.converged?'ĐÃ HỘI TỤ':'CHƯA HỘI TỤ','trạng thái']])+'</div><div class="flow-focus '+(frame.converged?'good':'warn')+'"><b>'+(frame.converged?'✓ Dừng K-means':'→ Sang vòng tiếp theo')+'</b>'+esc(frame.convergence_reason||frame.note||'')+'</div><div class="small">Lần khởi tạo '+run.run+(run.best?' là lần thắng theo inertia của KMeans.fit().':'; đây chưa phải lần có inertia nhỏ nhất.')+' Nếu chưa hội tụ, thuật toán tiếp tục gán lại câu với tâm mới.</div>';
}
function kmBestStep(){const runs=DATA.kmeans.restarts||[];const rows=runs.map(r=>'<tr class="'+(r.best?'won':'')+'"><td>Lần '+r.run+'</td><td>'+r.seed_indices.map(i=>esc(DATA.sentences[i]?.id||('C'+(i+1))).replaceAll('</td>','')).join(', ')+'</td><td class="num">'+num(r.final_inertia,9)+'</td><td>'+(r.best?'<span class="pill">★ Chọn</span>':'Không chọn')+'</td></tr>');return '<div class="flow-equation">KMeans.fit() chạy từng lần khởi tạo, sau đó chọn kết quả có inertia nhỏ nhất.</div>'+rowTable(['Lần khởi tạo','Seed câu','Inertia cuối thực tế','Quyết định'],rows)+'<div class="flow-result"><b>Kết quả được dùng cho bước sau: lần '+DATA.kmeans.best_run+'</b><br>Inertia = '+num((runs.find(r=>r.best)||{}).final_inertia,9)+'.</div>'+(DATA.kmeans.trace_matches_sklearn?'<div class="note">Trace của lần thắng khớp với nhãn/tâm cuối của KMeans.fit().</div>':'<div class="note warn">Có sai khác giữa trace trình bày và kết quả cuối của sklearn; kết quả thật từ KMeans.fit() được ưu tiên.</div>');}
function kmAnimationRun(){
  const runs=DATA.kmeans.restarts||[];
  return runs.find(r=>r.best)||runs.find(r=>Number(r.run)===Number(DATA.kmeans.best_run))||runs[0]||{history:DATA.kmeans.history||[],seed_indices:[],run:1,best:true};
}
function kmSceneFrame(item){
  const run=runFor(item),history=run.history||DATA.kmeans.history||[];
  return history[item.frameIndex]||history[0]||null;
}
function kmSvgBase(frame,phase,cursor,pointPhase='focus'){
  const pts=DATA.sentences||[],oldCenters=(frame?.centers||[]),after=(frame?.centers_after||oldCenters);
  const allX=pts.map(p=>Number(p.x)).concat(oldCenters.map(c=>Number(c.x)),after.map(c=>Number(c.x))).filter(Number.isFinite);
  const allY=pts.map(p=>Number(p.y)).concat(oldCenters.map(c=>Number(c.y)),after.map(c=>Number(c.y))).filter(Number.isFinite);
  const px=52,py=42,pw=660,ph=330;
  const project=pcaProjector(allX,allY,px,py,pw,ph),X=project.x,Y=project.y;
  const centers=phase==='complete'?after:oldCenters;
  const shownCenters=phase==='init-progress'?oldCenters.slice(0,Math.max(0,cursor+1)):centers;
  const assignments=(phase==='complete'?(frame?.post_assignments||frame?.assignments):frame?.assignments)||[];
  const matrix=frame?.distance_matrix||[];
  let out='<svg class="km-lesson-svg" viewBox="0 0 760 420" role="img" aria-label="Trực quan hóa K-Means: điểm câu, tâm cụm, khoảng cách và cập nhật tâm">';
  out+='<defs><marker id="km-arrow-new" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#8B7DD0"/></marker></defs>';
  for(let g=0;g<5;g++){
    const gx=px+pw*g/4,gy=py+ph*g/4;
    out+='<line x1="'+gx+'" y1="'+py+'" x2="'+gx+'" y2="'+(py+ph)+'" stroke="#F0F2F8" stroke-width=".8"/>';
    out+='<line x1="'+px+'" y1="'+gy+'" x2="'+(px+pw)+'" y2="'+gy+'" stroke="#F0F2F8" stroke-width=".8"/>';
  }
  out+='<text x="16" y="20" fill="#8790A3" font-size="10">PCA 2D · độ dài đường phản ánh hình chiếu; gán cụm dùng vector gốc 256D</text>';
  if(phase==='sweep'&&pts[cursor]&&['measure','assigned'].includes(pointPhase)){
    const p=pts[cursor],si=Number(p.index??cursor),cx=X(p.x),cy=Y(p.y),chosen=Number(assignments[si]??0),distances=matrix[si]||[],revealChoice=pointPhase==='assigned';
    centers.forEach((c,j)=>{
      const color=KM_COLORS[j%KM_COLORS.length],isChosen=j===chosen,highlight=revealChoice&&isChosen,tx=X(c.x),ty=Y(c.y);
      const dx=(cx+tx)/2,dy=(cy+ty)/2+((j%4)<2?-11:13);
      const delay=j*48,projectedDistance=Math.hypot(Number(p.x)-Number(c.x),Number(p.y)-Number(c.y));
      const trueDistance=Number.isFinite(Number(distances[j]))?Math.sqrt(Math.max(0,Number(distances[j]))):NaN;
      const trueDistanceText=Number.isFinite(trueDistance)?num(trueDistance,5):'chưa lưu';
      const edgeLabel=Number.isFinite(trueDistance)?num(trueDistance,2):'—';
      out+='<line class="km-distance-line" x1="'+cx+'" y1="'+cy+'" x2="'+tx+'" y2="'+ty+'" stroke="'+(highlight?color:'#CBD1DC')+'" stroke-width="'+(highlight?'2.1':'.85')+'" opacity="'+(highlight?'.95':'.4')+'" style="animation-delay:'+delay+'ms"><title>Khoảng cách thật Word2Vec 256D = '+trueDistanceText+'; khoảng cách PCA 2D = '+num(projectedDistance,5)+(highlight?' · được chọn theo khoảng cách thật 256D':'')+'</title></line>';
      out+='<text class="km-distance-label" x="'+dx+'" y="'+dy+'" text-anchor="middle" style="animation-delay:'+(delay+80)+'ms">d = '+edgeLabel+'</text>';
    });
  }
  const labelOffsets=[{dx:9,dy:-8,anchor:'start'},{dx:9,dy:13,anchor:'start'},{dx:-9,dy:-8,anchor:'end'},{dx:-9,dy:13,anchor:'end'},{dx:0,dy:-13,anchor:'middle'},{dx:0,dy:18,anchor:'middle'}];
  pts.forEach((p,i)=>{
    const si=Number(p.index??i),cl=Number(assignments[si]??0),current=phase==='sweep'&&i===cursor;
    const assignedNow=current&&pointPhase==='assigned';
    const processed=phase==='complete'||phase==='update'||(phase==='sweep'&&(i<cursor||assignedNow));
    const color=KM_COLORS[cl%KM_COLORS.length],fill=processed?color:'#C6CDD8',x=X(p.x),y=Y(p.y),label=labelOffsets[i%labelOffsets.length];
    const pointClass=(phase==='points'?' km-point-enter':'')+(assignedNow?' km-assigned-now':'')+(phase==='sweep'&&!current?' km-point-muted':'')+(current?' km-point-current':'');
    const labelClass='km-point-label'+(phase==='sweep'&&!current?' km-point-label-muted':'')+(current?' km-point-label-current':'');
    const delay=phase==='points'?Math.min(i*9,420):0;
    if(current)out+='<circle class="km-current-ring" cx="'+x+'" cy="'+y+'" r="12"/>';
    out+='<circle class="km-dot'+pointClass+'" style="--cluster-color:'+color+';animation-delay:'+delay+'ms" cx="'+x+'" cy="'+y+'" r="'+(current?'6':'4.2')+'" fill="'+fill+'" stroke="'+(current?'#E6C98E':'#fff')+'" stroke-width="'+(current?'2':'1.1')+'"><title>'+esc(p.id)+' · '+(processed?'Cụm '+(cl+1):'chưa gán cụm')+'</title></circle>';
    out+='<text class="'+labelClass+'" x="'+(x+label.dx)+'" y="'+(y+label.dy)+'" text-anchor="'+label.anchor+'" fill="'+(current?'#8B7958':'#758096')+'" font-size="'+(current?'10':'8')+'" font-weight="'+(current?'900':'600')+'">'+esc(p.id)+'</text>';
  });
  shownCenters.forEach((c,j)=>{
    const x=X(c.x),y=Y(c.y),color=KM_COLORS[j%KM_COLORS.length],isFresh=phase==='init-progress'&&j===cursor;
    if(phase==='update'&&after[j]){
      const nx=X(after[j].x),ny=Y(after[j].y);
      out+='<line class="km-center-move" x1="'+x+'" y1="'+y+'" x2="'+nx+'" y2="'+ny+'" stroke="'+color+'" stroke-width="1.7" marker-end="url(#km-arrow-new)"/>';
      out+='<circle class="km-center-old" cx="'+x+'" cy="'+y+'" r="12" fill="white" stroke="'+color+'" stroke-width="1.8"><title>Tâm cũ '+(j+1)+'</title></circle>';
      out+='<path d="'+starPath(nx,ny,9.5,4.1)+'" fill="'+color+'" opacity=".22" stroke="'+color+'" stroke-width="1.1" stroke-dasharray="3 3"><title>Vị trí mới của tâm '+(j+1)+'</title></path>';
      out+='<g class="km-center-new-group"><animateTransform attributeName="transform" type="translate" from="'+x+' '+y+'" to="'+nx+' '+ny+'" dur="1.15s" calcMode="spline" keySplines="0.22 1 0.36 1" fill="freeze"/><path class="km-center-moving-star" d="'+starPath(0,0,9.5,4.1)+'" fill="'+color+'" stroke="white" stroke-width="1.4"><title>Tâm mới '+(j+1)+'</title></path><text x="11" y="-9" fill="#68728A" font-size="10" font-weight="800">'+(j+1)+' mới</text></g>';
      out+='<text x="'+(x+11)+'" y="'+(y+15)+'" fill="'+color+'" font-size="10" opacity=".7">'+(j+1)+' cũ</text>';
    }else{
      if(isFresh)out+='<circle class="km-seed-pulse" cx="'+x+'" cy="'+y+'" r="14"/>';
      out+='<path class="'+(isFresh?'km-center-new':'km-center-still')+'" d="'+starPath(x,y,9.5,4.1)+'" fill="'+color+'" stroke="white" stroke-width="1.4"><title>Tâm cụm '+(j+1)+'</title></path>';
      out+='<text x="'+(x+13)+'" y="'+(y-10)+'" fill="#68728A" font-size="10" font-weight="800">'+(j+1)+'</text>';
    }
  });
  out+='</svg><div class="km-legend"><span><i></i>Câu chưa gán</span><span><i style="background:'+KM_COLORS[0]+'"></i>Màu cụm được chọn</span><span><i class="center"></i>Tâm cụm ★</span><span><i class="old"></i>Vị trí tâm cũ</span><span>Nhãn d = khoảng cách thật trong vector Word2Vec 256D</span><span>Độ dài hình học của cạnh vẫn là hình chiếu PCA 2D.</span></div>';
  return out;
}
function kmMovementSvg(frame, activeCluster){
  const pts=DATA.sentences||[],oldCenters=frame.centers||[],newCenters=frame.centers_after||oldCenters;
  const k=Math.max(oldCenters.length,newCenters.length),assign=frame.assignments||[];
  if(!k)return '<div class="note warn">Không có tâm cụm để vẽ.</div>';
  const allX=pts.map(p=>Number(p.x)).concat(oldCenters.map(c=>Number(c.x)),newCenters.map(c=>Number(c.x)));
  const allY=pts.map(p=>Number(p.y)).concat(oldCenters.map(c=>Number(c.y)),newCenters.map(c=>Number(c.y)));
  const X=pcaProjector(allX,allY,52,43,660,325).x,Y=pcaProjector(allX,allY,52,43,660,325).y;
  const j=Math.max(0,Math.min(activeCluster,k-1)),old=oldCenters[j],next=newCenters[j]||old;
  const color=KM_COLORS[j%KM_COLORS.length];
  let out='<svg class="km-lesson-svg" viewBox="0 0 760 420" role="img" aria-label="Dịch chuyển riêng tâm cụm '+(j+1)+'">';
  for(let g=0;g<5;g++){
    const gx=52+660*g/4,gy=43+325*g/4;
    out+='<line x1="'+gx+'" y1="43" x2="'+gx+'" y2="368" stroke="#F0F2F8" stroke-width=".8"/><line x1="52" y1="'+gy+'" x2="712" y2="'+gy+'" stroke="#F0F2F8" stroke-width=".8"/>';
  }
  out+='<text x="16" y="20" fill="#6E7890" font-size="10">Vòng '+esc(frame.iteration??'—')+' · tâm đang xem '+(j+1)+' / '+k+' · hình chiếu PCA 2D</text>';
  // Plot sentence points; emphasise members of the selected cluster and mute other clusters.
  pts.forEach((p,i)=>{
    const cl=Number(assign[Number(p.index??i)]??0),selected=cl===j,x=X(p.x),y=Y(p.y),c=KM_COLORS[cl%KM_COLORS.length];
    out+='<circle cx="'+x+'" cy="'+y+'" r="'+(selected?'5':'3.4')+'" fill="'+c+'" opacity="'+(selected?'.95':'.20')+'" stroke="#fff" stroke-width="1"><title>'+esc(p.id+' · cụm '+(cl+1))+'</title></circle>';
    if(selected)out+='<text x="'+(x+8)+'" y="'+(y-8)+'" font-size="9" font-weight="700" fill="#263653" paint-order="stroke" stroke="#fff" stroke-width="3">'+esc(p.id)+'</text>';
  });
  // Other centers stay as muted reference marks; only the active center is animated.
  oldCenters.forEach((c,idx)=>{
    if(idx===j)return;const x=X(c.x),y=Y(c.y),cc=KM_COLORS[idx%KM_COLORS.length];
    out+='<path d="'+starPath(x,y,6,2.6)+'" fill="'+cc+'" opacity=".25"/><text x="'+(x+8)+'" y="'+(y-7)+'" font-size="9" fill="#8790A3" opacity=".55">'+(idx+1)+'</text>';
  });
  if(old&&next){
    const x=X(old.x),y=Y(old.y),nx=X(next.x),ny=Y(next.y),distance=Math.hypot(Number(next.x)-Number(old.x),Number(next.y)-Number(old.y));
    out+='<defs><marker id="km-move-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="'+color+'"/></marker></defs>';
    if(distance>1e-9)out+='<line class="km-center-move" x1="'+x+'" y1="'+y+'" x2="'+nx+'" y2="'+ny+'" stroke="'+color+'" stroke-width="2.4" marker-end="url(#km-move-arrow)" opacity=".95"><title>Tâm dịch từ vị trí cũ tới trung bình vector mới</title></line>';
    out+='<circle cx="'+x+'" cy="'+y+'" r="12" fill="#fff" stroke="'+color+'" stroke-width="2" stroke-dasharray="4 3" opacity=".75"><title>Tâm cũ '+(j+1)+'</title></circle>';
    out+='<path d="'+starPath(nx,ny,13,5.2)+'" fill="'+color+'" stroke="#fff" stroke-width="1.8"><title>Tâm mới '+(j+1)+'</title></path>';
    out+='<text x="'+(x+13)+'" y="'+(y+18)+'" font-size="10" font-weight="800" fill="#5D6882" paint-order="stroke" stroke="#fff" stroke-width="3">'+(j+1)+' cũ</text>';
    out+='<text x="'+(nx+14)+'" y="'+(ny-12)+'" font-size="11" font-weight="900" fill="#202B50" paint-order="stroke" stroke="#fff" stroke-width="3.5">'+(j+1)+' mới</text>';
  }
  return out+'</svg><div class="km-legend"><span><i style="background:'+color+'"></i>Điểm thuộc cụm '+(j+1)+'</span><span><i class="old"></i>Tâm cũ</span><span><i class="center"></i>Tâm mới</span><span>Nét màu chỉ hướng dịch chuyển của tâm đang xét.</span></div>';
}
function kmMoveRefresh(){
  if(!kmMoveState)return;const st=kmMoveState,host=$('km-move-stage'),centers=st.frame.centers_after||st.frame.centers||[],k=centers.length;
  if(!host)return;
  host.innerHTML=kmMovementSvg(st.frame,st.clusterIndex);
  const u=(st.frame.cluster_updates||[]).find(x=>Number(x.cluster)===Number(st.clusterIndex));
  const shift=Number(u?.shift_sq??0),status=$('km-move-status'),metric=$('km-move-metric'),title=$('km-move-title'),body=$('km-move-text');
  if(status)status.textContent='DỊCH TÂM μ'+(st.clusterIndex+1)+' / '+k;
  if(metric)metric.textContent='μ'+(st.clusterIndex+1)+' · '+num(shift,8);
  if(title)title.textContent='Tâm cụm '+(st.clusterIndex+1)+': vị trí cũ → vị trí mới';
  if(body)body.textContent=shift>1e-12?'Tâm mới được tính bằng trung bình vector các câu đã gán vào cụm này. Đường màu chỉ biểu diễn hướng dịch chuyển trên hình chiếu PCA; phép tính thực tế dùng vector 256 chiều.':'Tâm này gần như không dịch chuyển ở vòng hiện tại (dịch tâm² = '+num(shift,10)+').';
  const prev=$('km-move-prev'),next=$('km-move-next'),play=$('km-move-play');
  if(prev)prev.disabled=st.clusterIndex<=0;
  if(next)next.textContent=st.clusterIndex>=k-1?'Hoàn tất bước →':'Tâm tiếp →';
  if(play)play.textContent=st.running?'⏸ Tạm dừng':(st.clusterIndex>=k-1?'↻ Xem lại tâm':'▶ Tự xem lần lượt');
}
function kmMoveStopTimer(){if(kmMoveTimer){clearTimeout(kmMoveTimer);kmMoveTimer=null;}}
function kmMoveTick(){
  const st=kmMoveState;if(!st||!st.running)return;
  const k=(st.frame.centers_after||st.frame.centers||[]).length;
  if(st.clusterIndex<k-1){st.clusterIndex++;kmMoveRefresh();kmMoveTimer=setTimeout(kmMoveTick,1050);}
  else{st.running=false;kmMoveTimer=null;kmMoveRefresh();}
}
function kmMovePlay(){
  if(!kmMoveState)return;const st=kmMoveState;
  if(st.running){st.running=false;kmMoveStopTimer();kmMoveRefresh();return;}
  if(st.clusterIndex>=(st.frame.centers_after||st.frame.centers||[]).length-1)st.clusterIndex=0;
  st.running=true;kmMoveRefresh();kmMoveTimer=setTimeout(kmMoveTick,500);
}
function kmMovePrev(){if(!kmMoveState)return;kmMoveState.running=false;kmMoveStopTimer();kmMoveState.clusterIndex=Math.max(0,kmMoveState.clusterIndex-1);kmMoveRefresh();}
function kmMoveNext(){
  if(!kmMoveState)return;const st=kmMoveState;st.running=false;kmMoveStopTimer();
  const k=(st.frame.centers_after||st.frame.centers||[]).length;
  if(st.clusterIndex<k-1){st.clusterIndex++;kmMoveRefresh();}
  else flowGo(flowIndex+1);
}
function kmMovementStep(item){
  const frame=kmSceneFrame(item);if(!frame)return '<div class="note warn">Không có nhật ký dịch chuyển tâm ở vòng này.</div>';
  const k=(frame.centers_after||frame.centers||[]).length;
  const rows=(frame.cluster_updates||[]).map(u=>'<tr><td>μ'+(Number(u.cluster)+1)+'</td><td>'+Number(u.count||0)+'</td><td class="num">'+num(u.shift_sq,9)+'</td><td class="small">'+((u.members||[]).map(i=>esc(DATA.sentences[i]?.id||('C'+(i+1)))).join(', ')||'—')+'</td></tr>');
  return '<div class="km-lesson-shell"><div class="km-lesson-toolbar"><button type="button" id="km-move-prev">← Tâm trước</button><button type="button" class="km-primary" id="km-move-play">▶ Tự xem lần lượt</button><button type="button" id="km-move-next">Tâm tiếp →</button></div><div class="km-lesson-status" id="km-move-status">Dịch tâm μ1 / '+k+'</div><div class="km-lesson-stage" id="km-move-stage">'+kmMovementSvg(frame,0)+'</div><div class="km-lesson-explain"><div><b id="km-move-title">Tâm cụm 1: vị trí cũ → vị trí mới</b><p id="km-move-text">Tâm mới là trung bình các vector của những câu được gán vào cụm. Đường màu biểu diễn hướng dịch chuyển trên PCA 2D.</p></div><div class="km-lesson-metric" id="km-move-metric">μ1</div></div></div><div class="flow-equation">Cập nhật tâm: <span class="math-equation">μ<sub>j</sub><sup>(t+1)</sup> = (1 / |C<sub>j</sub>|) ∑<sub>xᵢ ∈ Cⱼ</sub> xᵢ</span><br>Chỉ biểu diễn một tâm tại mỗi lượt bấm để dễ quan sát. Bấm “Tâm tiếp” để lần lượt xem đủ '+k+' tâm.</div><h3>Nhật ký dịch chuyển của các tâm vòng '+esc(frame.iteration??item.frameIndex)+'</h3>'+table(['Tâm','Số câu dùng để cập nhật','Tổng dịch tâm²','Các câu thuộc cụm'],rows)+'<div class="metric-row">'+metrics([[num(frame.shift,8),'tổng dịch tâm² trong vòng'],[num(frame.inertia,7),'inertia'],[frame.converged?'ĐÃ HỘI TỤ':'TIẾP TỤC','trạng thái sau vòng']])+'</div><div class="flow-result '+(frame.converged?'good':'')+'"><b>'+(frame.converged?'✓ Đạt điều kiện dừng':'→ Chuyển sang vòng Lloyd tiếp theo')+'</b><br>'+esc(frame.convergence_reason||'')+'</div><div class="note">Độ dịch chuyển được tính từ vector 256 chiều. Khoảng dịch trên sơ đồ chỉ là hình chiếu PCA 2D.</div>';
}
function bindKmMoveControls(item){
  kmMoveStopTimer();const frame=kmSceneFrame(item);if(!frame)return;
  kmMoveState={item,frame,clusterIndex:0,running:false};
  $('km-move-prev')?.addEventListener('click',kmMovePrev);$('km-move-play')?.addEventListener('click',kmMovePlay);$('km-move-next')?.addEventListener('click',kmMoveNext);
  kmMoveRefresh();
}

function kmPointsStep(){
  const pts=DATA.sentences||[];
  return '<div class="km-lesson-shell"><div class="km-lesson-stage">'+kmSvgBase(null,'points',-1)+'</div><div class="km-lesson-explain"><div><b>Các điểm câu</b><p>Chấm tròn là câu; vị trí là PCA 2D để dễ nhìn.</p></div><div class="km-lesson-metric">'+pts.length+' điểm</div></div></div>';
}
function kmInitStep(){
  const run=kmAnimationRun(),frame=(run.history||DATA.kmeans.history||[])[0];
  if(!frame)return '<div class="note warn">Không có trạng thái khởi tạo tâm để hiển thị.</div>';
  const k=(frame.centers||[]).length;
  return '<div class="km-lesson-shell"><div class="km-lesson-toolbar"><button type="button" id="km-init-prev">← Tâm trước</button><button type="button" class="km-primary" id="km-init-play">⏸ Tạm dừng</button><button type="button" id="km-init-next">Tâm tiếp →</button></div><div class="km-lesson-status" id="km-init-status">Các điểm đã hiện · chưa chọn tâm</div><div class="km-lesson-stage" id="km-init-stage">'+kmSvgBase(frame,'init-progress',-1)+'</div><div class="km-lesson-explain"><div><b>Khởi tạo K-means++</b><p>Hình thoi là tâm được chọn lần lượt. Tâm đầu chọn ngẫu nhiên; các tâm sau ưu tiên vùng cách xa tâm đã có.</p></div><div class="km-lesson-metric" id="km-init-metric">0 / '+k+'</div></div></div>';
}
function kmTeachingFrameStep(item){
  const frame=kmSceneFrame(item);if(!frame)return '<div class="note warn">Không có dữ liệu cho vòng K-Means này.</div>';
  return '<div class="km-lesson-shell"><div class="km-lesson-toolbar"><button type="button" id="km-lesson-prev">← Trước</button><button type="button" class="km-primary" id="km-lesson-play">⏸ Tạm dừng</button><button type="button" id="km-lesson-next">Tiếp →</button></div><div class="km-lesson-status" id="km-lesson-status">Chuẩn bị điểm đầu tiên…</div><div class="km-lesson-stage" id="km-lesson-stage">'+kmSvgBase(frame,'sweep',0,'focus')+'</div><div class="km-lesson-explain"><div><b id="km-lesson-explain-title">Duyệt từng điểm</b><p id="km-lesson-explain-text">Điểm đang xét sẽ nối tới các tâm, hiện khoảng cách rồi đổi màu theo tâm gần nhất.</p></div><div class="km-lesson-metric" id="km-lesson-metric">1 / '+DATA.sentences.length+'</div></div></div><div class="flow-equation">Gán cụm: <span class="math-equation">cᵢ = arg min<sub>j</sub> d²(xᵢ, μⱼ)</span><br>Ở bước này chỉ xem việc gán câu. Việc cập nhật và dịch chuyển tâm được tách sang bước tiếp theo.</div><div class="flow-focus"><b>Điều cần quan sát</b>Câu được tô màu theo tâm gần nhất dựa trên khoảng cách của vector Word2Vec 256 chiều; hình vẽ PCA 2D chỉ dùng để minh họa.</div>';
}
function kmFrameHasMovement(frame){return Number(frame?.shift||0)>1e-12||(frame?.cluster_updates||[]).some(u=>Number(u.shift_sq||0)>1e-12);}
function kmLessonStopTimer(){if(kmLessonTimer){clearTimeout(kmLessonTimer);kmLessonTimer=null;}}
function kmLessonRefresh(){
  if(!kmLessonState)return;const st=kmLessonState,host=$('km-lesson-stage');if(!host)return;
  host.innerHTML=kmSvgBase(st.frame,st.phase,st.cursor,st.pointPhase||'focus');
  const total=(DATA.sentences||[]).length,idx=Math.max(0,Math.min(st.cursor,total-1)),p=DATA.sentences[idx],cluster=Number(st.frame.assignments?.[Number(p?.index??idx)]??0);
  let title='',text='',status='',metric='';
  if(st.phase==='sweep'){
    if(st.pointPhase==='focus'){
      title='Điểm đang xét: '+(p?.id||('điểm '+(idx+1)));
      text='Chọn điểm; chưa gán màu cụm.';
      status='ĐIỂM '+(idx+1)+' / '+total+' · CHỌN ĐIỂM';
    }else if(st.pointPhase==='measure'){
      title='Khoảng cách tới từng tâm';
      text='Nhãn PCA là khoảng cách trên hình chiếu; bảng bên dưới là khoảng cách thật 256D dùng để chọn tâm gần nhất.';
      status='ĐIỂM '+(idx+1)+' / '+total+' · SO KHOẢNG CÁCH';
    }else{
      const dist=Math.sqrt(Math.max(0,Number((st.frame.distance_matrix?.[Number(p?.index??idx)]||[])[cluster]||0)));
      title=(p?.id||('Điểm '+(idx+1)))+' → Cụm '+(cluster+1);
      text='Chọn tâm gần nhất theo khoảng cách thật 256D (d = '+num(dist,4)+'); điểm đổi sang màu cụm, dù đường trên PCA có thể không ngắn nhất.';
      status='ĐIỂM '+(idx+1)+' / '+total+' · ĐÃ GÁN CỤM';
    }
    metric=(idx+1)+' / '+total;
  }else if(st.phase==='update'){
    title='Tâm cụm di chuyển';text='Vị trí cũ mờ dần; tâm mới hiện ở đích rồi trượt tới trung bình của cụm.';
    status='CẬP NHẬT TÂM';metric='μ cũ → μ mới';
  }else{
    title='Đã gán cụm cho toàn bộ điểm';
    text='Bước gán cụm đã hoàn tất. Sang bước kế tiếp để xem riêng từng tâm dịch chuyển từ vị trí cũ đến vị trí mới.';
    status='✓ XONG GÁN CỤM';metric=total+' / '+total;
  }
  if($('km-lesson-explain-title'))$('km-lesson-explain-title').textContent=title;
  if($('km-lesson-explain-text'))$('km-lesson-explain-text').textContent=text;
  if($('km-lesson-status'))$('km-lesson-status').textContent=status;
  if($('km-lesson-metric'))$('km-lesson-metric').textContent=metric;
  if(typeof renderFlowStrip==='function')renderFlowStrip(st.item);
  const prev=$('km-lesson-prev'),next=$('km-lesson-next'),play=$('km-lesson-play');
  if(prev)prev.disabled=st.phase==='sweep'&&st.cursor===0&&st.pointPhase==='focus';
  if(next){
    if(st.phase==='sweep')next.textContent=st.pointPhase==='focus'?'Khoảng cách →':(st.pointPhase==='measure'?'Gán màu →':(st.cursor<total-1?'Điểm tiếp →':'Xong gán cụm →'));
    else next.textContent='Sang bước dịch tâm →';
  }
  if(play)play.textContent=st.running?'⏸ Tạm dừng':(st.phase==='complete'?'↻ Chạy lại':'▶ Chạy hoạt ảnh');
}
function kmLessonTick(){
  const st=kmLessonState;if(!st||!st.running)return;
  if(st.phase==='sweep'){
    if(st.pointPhase==='focus')st.pointPhase='measure';
    else if(st.pointPhase==='measure')st.pointPhase='assigned';
    else if(st.cursor<(DATA.sentences||[]).length-1){st.cursor++;st.pointPhase='focus';}
    else st.phase='complete';
  }else if(st.phase==='update')st.phase='complete';
  else{st.running=false;kmLessonTimer=null;kmLessonRefresh();return;}
  kmLessonRefresh();
  const delay=st.phase==='update'?1350:(st.phase==='complete'?650:(st.pointPhase==='measure'?680:(st.pointPhase==='assigned'?430:250)));
  kmLessonTimer=setTimeout(kmLessonTick,delay);
}
function kmLessonPlay(){
  if(!kmLessonState)return;const st=kmLessonState;
  if(st.phase==='complete'){st.cursor=0;st.phase='sweep';st.pointPhase='focus';}
  if(st.running){st.running=false;kmLessonStopTimer();kmLessonRefresh();return;}
  st.running=true;kmLessonRefresh();kmLessonTimer=setTimeout(kmLessonTick,350);
}
function kmLessonPrev(){
  const st=kmLessonState;if(!st)return;st.running=false;kmLessonStopTimer();
  if(st.phase==='update'||st.phase==='complete'){
    st.phase='sweep';st.cursor=Math.max(0,(DATA.sentences||[]).length-1);st.pointPhase='assigned';
  }else if(st.phase==='sweep'){
    if(st.pointPhase==='assigned')st.pointPhase='measure';
    else if(st.pointPhase==='measure')st.pointPhase='focus';
    else if(st.cursor>0){st.cursor--;st.pointPhase='assigned';}
  }
  kmLessonRefresh();
}
function kmLessonNext(){
  const st=kmLessonState;if(!st)return;st.running=false;kmLessonStopTimer();
  if(st.phase==='sweep'){
    if(st.pointPhase==='focus')st.pointPhase='measure';
    else if(st.pointPhase==='measure')st.pointPhase='assigned';
    else if(st.cursor<(DATA.sentences||[]).length-1){st.cursor++;st.pointPhase='focus';}
    else st.phase='complete';
  }else if(st.phase==='update')st.phase='complete';
  else{flowGo(flowIndex+1);return;}
  kmLessonRefresh();
}
function bindKmLessonControls(item){
  kmLessonStopTimer();const frame=kmSceneFrame(item);if(!frame)return;
  kmLessonState={item,frame,cursor:0,phase:'sweep',pointPhase:'focus',running:true};
  $('km-lesson-play')?.addEventListener('click',kmLessonPlay);$('km-lesson-prev')?.addEventListener('click',kmLessonPrev);$('km-lesson-next')?.addEventListener('click',kmLessonNext);
  kmLessonRefresh();kmLessonTimer=setTimeout(kmLessonTick,260);
}
function kmInitStop(){if(kmInitTimer){clearTimeout(kmInitTimer);kmInitTimer=null;}}
function kmInitRefresh(){
  if(!kmInitState)return;const st=kmInitState,host=$('km-init-stage'),k=(st.frame.centers||[]).length;
  if(!host)return;
  host.innerHTML=kmSvgBase(st.frame,'init-progress',st.cursor);
  if($('km-init-status'))$('km-init-status').textContent=st.cursor<0?'Các điểm đã hiện · chưa có tâm':'Đã chọn tâm μ'+(st.cursor+1)+' / '+k;
  if($('km-init-metric'))$('km-init-metric').textContent=Math.max(0,st.cursor+1)+' / '+k;
  const prev=$('km-init-prev'),next=$('km-init-next'),play=$('km-init-play');
  if(prev)prev.disabled=st.cursor<0;
  if(next)next.textContent=st.cursor>=k-1?'Sang gán cụm →':'Tâm tiếp →';
  if(play)play.textContent=st.running?'⏸ Tạm dừng':(st.cursor>=k-1?'↻ Chọn lại':'▶ Tự chọn tâm');
}
function kmInitTick(){
  const st=kmInitState;if(!st||!st.running)return;
  const k=(st.frame.centers||[]).length;
  if(st.cursor<k-1){st.cursor++;kmInitRefresh();kmInitTimer=setTimeout(kmInitTick,560);}
  else{st.running=false;kmInitTimer=null;kmInitRefresh();}
}
function kmInitPlay(){
  if(!kmInitState)return;const st=kmInitState;
  if(st.running){st.running=false;kmInitStop();kmInitRefresh();return;}
  if(st.cursor>=(st.frame.centers||[]).length-1)st.cursor=-1;
  st.running=true;kmInitRefresh();kmInitTimer=setTimeout(kmInitTick,180);
}
function kmInitPrev(){
  if(!kmInitState)return;const st=kmInitState;st.running=false;kmInitStop();st.cursor=Math.max(-1,st.cursor-1);kmInitRefresh();
}
function kmInitNext(){
  if(!kmInitState)return;const st=kmInitState;st.running=false;kmInitStop();
  if(st.cursor<(st.frame.centers||[]).length-1){st.cursor++;kmInitRefresh();}
  else{flowGo(flowIndex+1);}
}
function bindKmInitControls(){
  kmInitStop();const run=kmAnimationRun(),frame=(run.history||DATA.kmeans.history||[])[0];if(!frame)return;
  kmInitState={frame,cursor:-1,running:true};
  $('km-init-play')?.addEventListener('click',kmInitPlay);$('km-init-prev')?.addEventListener('click',kmInitPrev);$('km-init-next')?.addEventListener('click',kmInitNext);
  kmInitRefresh();kmInitTimer=setTimeout(kmInitTick,350);
}

function cosineAngleSvg(score,id){
  const raw=Number(score),valid=Number.isFinite(raw),clamped=Math.max(-1,Math.min(1,valid?raw:0)),theta=Math.acos(clamped),degrees=theta*180/Math.PI;
  const ox=78,oy=190,len=145,ax=ox+len,ay=oy,bx=ox+len*Math.cos(theta),by=oy-len*Math.sin(theta),r=46,ex=ox+r*Math.cos(theta),ey=oy-r*Math.sin(theta);
  const large=theta>Math.PI?1:0;
  const arc=theta<0.001?'':('M '+(ox+r)+' '+oy+' A '+r+' '+r+' 0 '+large+' 0 '+ex+' '+ey);
  return '<div class="centroid-angle-card"><h3>Góc giữa vector câu và vector chủ đề</h3><p>Minh họa góc tương ứng với cosine thực tế của câu '+esc(id||'đang xét')+'. Độ dài hai mũi tên được chuẩn hóa để dễ nhìn hướng.</p><svg class="centroid-angle-svg" viewBox="0 0 560 250" role="img" aria-label="Góc cosine giữa vector câu và vector chủ đề"><defs><marker id="centroid-arrow-topic" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#765b9d"/></marker><marker id="centroid-arrow-sentence" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#3020B8"/></marker></defs><line x1="40" y1="'+oy+'" x2="520" y2="'+oy+'" stroke="#e2e8f2" stroke-width="1.5"/><line x1="'+ox+'" y1="35" x2="'+ox+'" y2="220" stroke="#e2e8f2" stroke-width="1.5"/><line class="angle-vector" x1="'+ox+'" y1="'+oy+'" x2="'+ax+'" y2="'+ay+'" stroke="#765b9d" stroke-width="5" marker-end="url(#centroid-arrow-topic)"/><line class="angle-vector" x1="'+ox+'" y1="'+oy+'" x2="'+bx+'" y2="'+by+'" stroke="#3020B8" stroke-width="5" marker-end="url(#centroid-arrow-sentence)"/><path class="angle-arc" d="'+arc+'" fill="none" stroke="#D9A36C" stroke-width="3"/><circle cx="'+ox+'" cy="'+oy+'" r="5" fill="#263653"/><text x="'+(ax-8)+'" y="'+(ay+24)+'" fill="#765b9d" font-size="13" font-weight="800" text-anchor="end">Vector chủ đề μ</text><text x="'+(bx+12)+'" y="'+(by-9)+'" fill="#3020B8" font-size="13" font-weight="800">Vector câu c</text><text x="'+(ox+60*Math.cos(theta/2))+'" y="'+(oy-60*Math.sin(theta/2)-5)+'" fill="#A47A4F" font-size="13" font-weight="900">θ = '+degrees.toFixed(1)+'°</text><text x="'+(ox+10)+'" y="'+(oy+20)+'" fill="#56627b" font-size="11">O</text><text x="280" y="232" text-anchor="middle" fill="#5f6c85" font-size="11">cos(c, μ) = (c · μ) / (||c|| × ||μ||) = '+num(clamped,6)+'</text></svg><div class="flow-equation">θ = arccos(cos(c, μ)) = arccos('+num(clamped,7)+') = <b>'+degrees.toFixed(2)+'°</b></div></div>';
}
function mmrTransferBoard(round){
  const q=DATA.mmr.pool[0],before=round.selected_before||[],remainingBefore=round.remaining_before||[],after=round.selected_after||[],remainingAfter=round.remaining_after||[];
  const winner=round.candidates?.find(c=>c.winner)||round.candidates?.find(c=>c.id===round.winner_id);
  const chip=(x,kind='')=>'<div class="mmr-transfer-chip '+kind+'"><b>'+esc(x.id)+'</b><small>'+esc((x.text||'').slice(0,118))+(String(x.text||'').length>118?'…':'')+'</small></div>';
  return '<div class="mmr-transfer-board"><div class="mmr-transfer-q"><b>Q · GIỮ NGUYÊN Q QUA MỌI VÒNG</b><p>'+esc(q?.id||'Q')+' · '+esc(q?.text||'')+'</p></div><div class="mmr-transfer-grid"><section class="mmr-transfer-col s-before"><h4>TRƯỚC LƯỢT · R ĐÃ CHỌN</h4>'+(before.map(x=>chip(x)).join('')||'<p class="small">∅</p>')+'<p class="small">Các câu dùng để tính độ trùng lặp Red.</p></section><section class="mmr-transfer-col r-before"><h4>ỨNG VIÊN CHƯA CHỌN</h4>'+(remainingBefore.map(x=>x.id===round.winner_id?chip(x,'winner'):chip(x,'remain')).join('')||'<p class="small">∅</p>')+'<div class="mmr-transfer-arrow">↓</div><p class="small">Câu viền xanh có MMRScore cao nhất; câu khác tiếp tục ở danh sách ứng viên.</p></section><section class="mmr-transfer-col s-after"><h4>SAU LƯỢT · R ∪ {winner}</h4>'+(after.map(x=>x.id===round.winner_id?chip(x,'arrive winner'):chip(x)).join('')||'<p class="small">∅</p>')+'<div class="mmr-transfer-arrow">←</div><p class="small">Câu thắng được chuyển vào tập đã chọn R.</p></section><section class="mmr-transfer-col r-after"><h4>ỨNG VIÊN CÒN LẠI</h4>'+(remainingAfter.map(x=>chip(x)).join('')||'<p class="small">Không còn ứng viên</p>')+'<p class="small">Câu thắng bị loại khỏi danh sách ứng viên; các câu còn lại được chấm lại ở lượt sau.</p></section></div><div class="mmr-transfer-equation">winner = argmax<sub>c ∈ U</sub> MMRScore(c)<br>R mới = R cũ ∪ {'+esc(round.winner_id)+'}<br>U mới = U cũ trừ {'+esc(round.winner_id)+'}</div></div>';
}

function kmFrameStep(item){
  const run=runFor(item),history=run.history||DATA.kmeans.history||[],frame=history[item.frameIndex];
  if(!frame)return '<div class="note warn">Không có dữ liệu cho vòng K-means này.</div>';
  const counts=Array.from({length:DATA.kmeans.k},(_,j)=>({cluster:j,members:DATA.sentences.filter((s,i)=>frame.assignments?.[s.index??i]===j)}));
  const clusters=counts.map(c=>'<section class="lecture-state" style="border-top:4px solid '+COLORS[c.cluster%COLORS.length]+'"><b>CỤM '+(c.cluster+1)+' · '+c.members.length+' câu</b><p>'+c.members.map(x=>'<span class="lecture-state-chip">'+esc(x.id)+'</span>').join(' ')+'</p></section>').join('');
  const updateRows=(frame.cluster_updates||[]).map(u=>'<tr><td><b>Cụm '+(u.cluster+1)+'</b></td><td>'+u.count+'</td><td>'+((u.members||[]).map(i=>esc(DATA.sentences[i]?.id||('C'+(i+1)))).join(', ')||'—')+'</td><td class="num">'+num(u.shift_sq,7)+'</td><td class="num">['+(u.new_preview||[]).slice(0,5).map(v=>num(v,3)).join(', ')+(u.new_preview?.length>5?', …':'')+']</td></tr>');
  return '<div class="flow-focus '+(frame.converged?'good':'')+'"><b>'+(item.frameIndex===0?'Bước khởi tạo':('Vòng '+(frame.iteration??item.frameIndex)))+'</b>'+esc(frame.note||frame.convergence_reason||'Mỗi câu được gán theo tâm gần nhất; các tâm được cập nhật sau khi gán cụm.')+'</div>'+plot(frame)+'<div class="metric-row">'+metrics([[item.frameIndex===0?'Khởi tạo':String(frame.iteration??item.frameIndex),'trạng thái'],[num(frame.shift,7),'tổng dịch tâm²'],[num(frame.inertia,7),'inertia'],[frame.converged?'ĐÃ HỘI TỤ':'TIẾP TỤC','điều kiện dừng']])+'</div><div class="lecture-equation">Gán cụm: cᵢ = argminⱼ d²(xᵢ, μⱼ)<br>Cập nhật tâm: μⱼ mới = trung bình vector của các câu thuộc cụm j</div>'+flowCaption('Sau bước này, các câu đang nằm ở cụm nào?','Màu trong biểu đồ tương ứng với màu của cụm')+'<div class="lecture-state-columns">'+clusters+'</div>'+(updateRows.length?flowCaption('Tâm cụm sau cập nhật','Hiển thị một phần vector để dễ đọc')+rowTable(['Cụm','Số câu','Thành viên','Dịch tâm²','5 chiều đầu của tâm mới'],updateRows):'<div class="note">Đây là trạng thái khởi tạo, chưa có phép cập nhật tâm mới.</div>')+'<div class="flow-result"><b>'+(frame.converged?'✓ K-means đã hội tụ ở trạng thái này.':'→ Tiếp tục sang trạng thái kế tiếp.')+'</b><br>'+esc(frame.convergence_reason||'')+'</div>';
}
function kmAnimationStep(){
  kmAnimIndex=0;
  const run=kmAnimationRun(),h=run.history||DATA.kmeans.history||[];
  const frameCount=Math.max(1,h.length);
  return '<div class="km-graph-player"><div class="km-graph-toolbar"><button type="button" id="km-graph-prev">← Trước</button><button type="button" id="km-graph-play" class="km-graph-play">▶ Phát</button><button type="button" id="km-graph-next">Tiếp →</button><button type="button" id="km-graph-reset">↺ Đầu</button></div><div id="km-chart-count" class="km-graph-status">Khởi tạo · 1 / '+frameCount+'</div><div id="km-chart-visual"></div></div><p class="small">Biểu đồ dùng PCA 2D để hiển thị; phép gán cụm dùng vector Word2Vec 256 chiều. Đang xem lần khởi tạo '+esc(run.run||DATA.kmeans.best_run||1)+(run.best?' (inertia thấp nhất).':'.')+'</p>';
}
function renderKmGraphOnly(){
  if(!$('km-chart-visual')||!DATA)return;
  const run=kmAnimationRun(),h=run.history||DATA.kmeans.history||[],last=Math.max(0,h.length-1);
  kmAnimIndex=Math.max(0,Math.min(kmAnimIndex,last));
  let chart='',label='';
  if(kmAnimIndex===0){
    const seeds=run.seed_indices||[];
    const seedItem={runIndex:Math.max(0,(DATA.kmeans.restarts||[]).indexOf(run)),seedPos:Math.max(0,seeds.length-1)};
    const firstFrame=h[0]||DATA.kmeans.history?.[0];
    chart=seeds.length?seedPlot(seedItem):(firstFrame?plot(firstFrame):'<div class="note warn">Không có lịch sử K-means để vẽ biểu đồ.</div>');
    label='Khởi tạo · '+(seeds.length?seeds.length+' tâm đã chọn':'tâm khởi tạo');
  }else{
    const frame=h[kmAnimIndex];
    chart=plot(frame);
    label='Vòng '+(frame.iteration??kmAnimIndex)+' · '+(frame.converged?'đã hội tụ':'đang cập nhật')+' · dịch tâm² '+num(frame.shift,7)+' · inertia '+num(frame.inertia,7);
  }
  $('km-chart-visual').innerHTML=chart;
  const count=$('km-chart-count');if(count)count.textContent=label+' · Khung '+(kmAnimIndex+1)+' / '+(last+1);
  const prev=$('km-graph-prev'),next=$('km-graph-next'),play=$('km-graph-play');
  if(prev)prev.disabled=kmAnimIndex===0;
  if(next)next.disabled=kmAnimIndex===last;
  if(play)play.textContent=kmAnimTimer?'⏸ Tạm dừng':'▶ Phát biểu đồ';
}
function stopKmGraph(){if(kmAnimTimer){clearTimeout(kmAnimTimer);kmAnimTimer=null;}const play=$('km-graph-play');if(play)play.textContent='▶ Phát biểu đồ';}
function playKmGraphNext(){kmAnimTimer=setTimeout(()=>{const run=kmAnimationRun(),last=Math.max(0,(run.history||DATA.kmeans.history||[]).length-1);if(kmAnimIndex>=last){stopKmGraph();return;}kmAnimIndex++;renderKmGraphOnly();playKmGraphNext();},Number($('flow-speed')?.value||1700));}
function bindKmGraphControls(){
  const prev=$('km-graph-prev'),next=$('km-graph-next'),reset=$('km-graph-reset'),play=$('km-graph-play');
  if(prev)prev.onclick=()=>{stopKmGraph();kmAnimIndex--;renderKmGraphOnly();};
  if(next)next.onclick=()=>{stopKmGraph();kmAnimIndex++;renderKmGraphOnly();};
  if(reset)reset.onclick=()=>{stopKmGraph();kmAnimIndex=0;renderKmGraphOnly();};
  if(play)play.onclick=()=>{if(kmAnimTimer){stopKmGraph();return;}const run=kmAnimationRun(),last=Math.max(0,(run.history||DATA.kmeans.history||[]).length-1);if(kmAnimIndex>=last)kmAnimIndex=0;renderKmGraphOnly();playKmGraphNext();};
  renderKmGraphOnly();
}

function representativeSummaryStep(){
  const steps=DATA.representative_steps||[];
  if(!steps.length)return '<div class="note warn">Không có kết quả chọn câu đại diện.</div>';
  const points=DATA.sentences||[];
  const inClusterCount=steps.filter(r=>Boolean(r.selected_in_cluster)).length;
  const clusterTables=steps.map((r,order)=>{
    // Each cluster table lists only sentences actually assigned to that cluster.
    const allCandidates=r.candidates||[];
    const candidates=allCandidates.filter(c=>Number(c.assigned_cluster)===Number(r.cluster)).slice().sort((a,b)=>
      (Number(a.distance)-Number(b.distance)) || (Number(a.index)-Number(b.index)));
    const rows=candidates.map((c,rank)=>{
      const selected=Number(c.index)===Number(r.selected_index)||Boolean(c.is_representative);
      const state=selected
        ? '<span class="rep-selected-pill">★ CÂU ĐƯỢC CHỌN</span>'
        : '<span class="rep-member-pill">Thuộc cụm</span>';
      return '<tr class="'+(selected?'rep-selected-row':'rep-member-row')+'">'+
        '<td class="num">'+(rank+1)+'</td>'+
        '<td><b>'+esc(c.id)+'</b></td>'+
        '<td class="sentence">'+esc(c.text||points[Number(c.index)]?.text||'')+'</td>'+
        '<td class="num"><b>'+num(Number(c.distance),7)+'</b></td>'+
        '<td>'+state+'</td></tr>';
    });
    const actualSelected=allCandidates.find(c=>Number(c.index)===Number(r.selected_index));
    const note=r.selected_in_cluster
      ? 'Câu được chọn thuộc cụm này và được đánh dấu ★ trong bảng.'
      : 'Câu pipeline thực tế chọn là '+esc(r.selected_id)+' (được gán vào cụm '+(Number(actualSelected?.assigned_cluster)+1)+'). Câu đó được hiển thị ở phần kết quả phía trên; bảng này chỉ liệt kê thành viên của cụm hiện tại.';
    return '<section class="rep-cluster-panel">'+
      '<div class="rep-cluster-head"><div><h3>Cụm '+(Number(r.cluster)+1)+' · tâm μ'+(Number(r.cluster)+1)+'</h3>'+
      '<p>Chỉ hiển thị các câu thuộc cụm này; khoảng cách Euclid thật tới tâm trong Word2Vec 256D.</p></div>'+
      '<span class="rep-count">'+candidates.length+' câu trong cụm</span></div>'+
      '<div class="rep-chosen-summary"><span class="rep-star">★</span><div><b>Đại diện pipeline chọn: '+esc(r.selected_id)+'</b><p>'+esc(r.selected_text||'')+'</p><small>Khoảng cách thật tới tâm này: '+num(Number(r.selected_distance),7)+'</small></div></div>'+
      table(['Hạng trong cụm','ID câu','Văn bản','Khoảng cách thật 256D','Trạng thái'],rows)+
      '<p class="rep-cluster-note">'+note+'</p></section>';
  }).join('');
  return metrics([[steps.length,'tâm cụm được xét'],[inClusterCount,'đại diện thuộc đúng cụm'],[points.length,'tổng số câu']])+
    '<div class="flow-focus"><b>Cách đọc bảng</b>Mỗi bảng chỉ liệt kê các câu được K-Means gán vào cụm tương ứng, sắp theo khoảng cách Euclid thật tới tâm từ nhỏ đến lớn. Câu được pipeline chọn được đánh dấu ★ nếu thuộc cụm đó.</div>'+clusterTables+
    (inClusterCount<steps.length?'<div class="note warn"><b>Lưu ý:</b> logic repository hiện tại có thể chọn câu gần một tâm nhất trên toàn bộ tập câu, dù câu đó được gán sang cụm khác. Khi đó câu đại diện thực tế vẫn hiện riêng ở phần trên, còn bảng chỉ chứa thành viên của cụm đang xem.</div>':'');
}
function representativeStep(item){
  const r=(DATA.representative_steps||[])[item.representativeIndex];
  if(!r)return '<div class="note warn">Không có nhật ký chọn đại diện.</div>';
  const frame=(DATA.kmeans.restarts||[]).find(x=>x.best)?.history?.slice(-1)[0]||DATA.kmeans.history.slice(-1)[0];
  const points=DATA.sentences||[],members=(r.candidates||[]).filter(c=>Number(c.assigned_cluster)===Number(r.cluster));
  const chosen=points[Number(r.selected_index)];
  const center=(frame?.centers_after||frame?.centers||[])[Number(r.cluster)];
  if(!center||!points.length)return '<div class="note warn">Thiếu tọa độ tâm hoặc điểm câu để vẽ.</div>';
  const visible=members.map(c=>points[Number(c.index)]).filter(Boolean);
  if(chosen&&!visible.some(p=>Number(p.index)===Number(chosen.index)))visible.push(chosen);
  const xs=visible.map(p=>Number(p.x)).concat(Number(center.x)),ys=visible.map(p=>Number(p.y)).concat(Number(center.y));
  const project=pcaProjector(xs,ys,70,55,760,350),X=project.x,Y=project.y;
  const color=COLORS[Number(r.cluster)%COLORS.length],cx=X(center.x),cy=Y(center.y);
  const edges=[];
  for(const c of members){
    const pt=points[Number(c.index)];if(!pt)continue;
    const x1=cx,y1=cy,x2=X(pt.x),y2=Y(pt.y);
    edges.push({c,pt,x1,y1,x2,y2,len:Math.hypot(x2-x1,y2-y1),pcaDistance:Math.hypot(Number(pt.x)-Number(center.x),Number(pt.y)-Number(center.y))});
  }
  let svg='<svg class="plot" viewBox="0 0 900 490" role="img" aria-label="Khoảng cách thật từ tâm cụm tới từng câu; tọa độ hiển thị bằng PCA 2D">';
  svg+='<text x="28" y="25" font-size="13" fill="#63708a">Nhãn cạnh: khoảng cách Euclid thật 256D · vị trí: PCA 2D</text>';
  // Draw the projected edges first. Their visual lengths are PCA lengths, but each label below
  // deliberately displays the true high-dimensional distance used by the pipeline.
  for(const e of edges){
    svg+='<line x1="'+e.x1+'" y1="'+e.y1+'" x2="'+e.x2+'" y2="'+e.y2+'" stroke="'+color+'" stroke-width="1.8" opacity=".72"><title>'+esc(e.c.id+' · khoảng cách Euclid thật 256D = '+num(Number(e.c.distance),9)+' · khoảng cách PCA 2D = '+num(e.pcaDistance,6))+'</title></line>';
  }
  // Reserve the visual area occupied by point labels and the centroid label so distance chips
  // can be moved away from them instead of stacking on top of the dense center region.
  const obstacles=[];
  const addObstacle=(x,y,w,h)=>obstacles.push({x,y,w,h});
  for(const c of members){
    const pt=points[Number(c.index)];if(!pt)continue;
    const x=X(pt.x),y=Y(pt.y),selected=Number(c.index)===Number(r.selected_index);
    addObstacle(x-13,y-15,Math.min(130,90+String(c.id||'').length*1.5),30);
    if(selected)addObstacle(x-17,y-17,34,34);
  }
  if(chosen&&!members.some(c=>Number(c.index)===Number(chosen.index))){
    const x=X(chosen.x),y=Y(chosen.y);addObstacle(x-13,y-15,110,30);addObstacle(x-17,y-17,34,34);
  }
  addObstacle(cx-18,cy-31,98,40);
  const intersects=(a,b,pad=3)=>!(a.x+a.w+pad<=b.x||b.x+b.w+pad<=a.x||a.y+a.h+pad<=b.y||b.y+b.h+pad<=a.y);
  const areaOverlap=(a,b)=>Math.max(0,Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x))*Math.max(0,Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y));
  const placed=[];let distanceLabels='';
  const labelW=82,labelH=21;
  // Place longer spokes first; labels on short, crowded spokes are nudged along/perpendicular
  // to the segment. Each chip keeps the cluster color and a tooltip with full precision.
  const labelEdges=edges.slice().sort((a,b)=>b.len-a.len);
  for(const e of labelEdges){
    const dx=e.x2-e.x1,dy=e.y2-e.y1,ln=Math.max(1,e.len),nx=-dy/ln,ny=dx/ln;
    let best=null,bestScore=Infinity;
    const ts=[.5,.38,.62,.25,.75,.12,.88,.04,.96];
    const offsets=[0,15,-15,27,-27,40,-40,54,-54,68,-68,82,-82];
    for(const t of ts){for(const off of offsets){
      let px=e.x1+dx*t+nx*off,py=e.y1+dy*t+ny*off;
      px=Math.max(46+labelW/2,Math.min(854-labelW/2,px));
      py=Math.max(43+labelH/2,Math.min(464-labelH/2,py));
      const box={x:px-labelW/2,y:py-labelH/2,w:labelW,h:labelH};
      let collision=0;
      for(const o of obstacles){if(intersects(box,o,3))collision+=12000+areaOverlap(box,o)*8;}
      for(const o of placed){if(intersects(box,o,4))collision+=16000+areaOverlap(box,o)*12;}
      const displacement=Math.hypot(px-(e.x1+dx*.5),py-(e.y1+dy*.5));
      const score=collision+displacement*.75+Math.abs(t-.5)*e.len*.32+Math.abs(off)*.13;
      if(score<bestScore){bestScore=score;best={x:px,y:py,box,t,off};}
    }}
    if(!best)continue;
    placed.push(best.box);
    const actual=Number(e.c.distance),label='d='+num(actual,5);
    // A light leader connects a displaced label to the corresponding edge, when needed.
    if(Math.abs(best.off)>11||Math.abs(best.t-.5)>.17){
      const ax=e.x1+dx*best.t,ay=e.y1+dy*best.t;
      distanceLabels+='<line x1="'+ax+'" y1="'+ay+'" x2="'+best.x+'" y2="'+best.y+'" stroke="'+color+'" stroke-width="1" opacity=".46" stroke-dasharray="2 3" pointer-events="none"/>'; 
    }
    distanceLabels+='<g class="centroid-distance-chip"><rect x="'+best.box.x+'" y="'+best.box.y+'" width="'+labelW+'" height="'+labelH+'" rx="6" fill="#fff" fill-opacity=".97" stroke="'+color+'" stroke-width="1.1" stroke-opacity=".72"/><text x="'+best.x+'" y="'+(best.y+3.5)+'" font-size="10" font-weight="800" fill="#263653" text-anchor="middle">'+esc(label)+'</text><title>'+esc(e.c.id+' · khoảng cách Euclid thật trong vector gốc 256D = '+num(actual,9)+' · khoảng cách PCA 2D = '+num(e.pcaDistance,6))+'</title></g>';
  }
  // Points and labels are drawn after the lines, so their markers remain crisp and readable.
  for(const c of members){
    const pt=points[Number(c.index)];if(!pt)continue;
    const selected=Number(c.index)===Number(r.selected_index),x=X(pt.x),y=Y(pt.y);
    if(selected)svg+='<circle cx="'+x+'" cy="'+y+'" r="15" fill="none" stroke="#D9A36C" stroke-width="3"/>';
    svg+='<circle cx="'+x+'" cy="'+y+'" r="'+(selected?9:6)+'" fill="'+(selected?'#D9A36C':color)+'" stroke="#fff" stroke-width="2"><title>'+esc(c.id+' · '+c.text)+'</title></circle>';
    svg+='<text x="'+(x+10)+'" y="'+(y+4)+'" font-size="11" font-weight="'+(selected?'900':'600')+'" fill="'+(selected?'#916A43':'#39445e')+'">'+esc(c.id)+(selected?' ★':'')+'</text>';
  }
  if(chosen&&!members.some(c=>Number(c.index)===Number(chosen.index))){
    const x=X(chosen.x),y=Y(chosen.y);
    svg+='<circle cx="'+x+'" cy="'+y+'" r="15" fill="none" stroke="#D9A36C" stroke-width="3"/><circle cx="'+x+'" cy="'+y+'" r="9" fill="#D9A36C" stroke="#fff" stroke-width="2"/><text x="'+(x+10)+'" y="'+(y+4)+'" font-size="11" font-weight="900" fill="#916A43">'+esc(r.selected_id)+' ★</text>';
  }
  svg+='<path d="'+starPath(cx,cy,13,5.5)+'" fill="'+color+'" stroke="#fff" stroke-width="1.5"/><text x="'+(cx+16)+'" y="'+(cy-10)+'" font-size="14" font-weight="900" fill="'+color+'">μ'+(Number(r.cluster)+1)+'</text>'+distanceLabels+'</svg>';
  svg+='<div class="legend"><span><i class="dot" style="background:'+color+'"></i>Điểm thuộc cụm '+(Number(r.cluster)+1)+'</span><span>★ tâm cụm</span><span>★ câu được pipeline chọn</span><span>Nhãn d trên cạnh = khoảng cách Euclid thật 256D; độ dài và tọa độ đường vẫn là hình chiếu PCA 2D.</span></div>';
  const outside=!r.selected_in_cluster;
  const distanceRows=members.map(c=>{const pt=points[Number(c.index)],pcaDistance=pt?Math.hypot(Number(pt.x)-Number(center.x),Number(pt.y)-Number(center.y)):NaN;return '<tr><td><b>'+esc(c.id)+'</b></td><td class="num">'+num(pcaDistance,5)+'</td><td class="num">'+num(c.distance,9)+'</td><td>'+(Number(c.index)===Number(r.selected_index)?'<span class="pill">★ Được chọn</span>':'')+'</td></tr>';});
  return '<div class="flow-equation math-equation"><span class="eq-line">i<sub>'+(Number(r.cluster)+1)+'</sub><sup>*</sup> = arg min<sub>i</sub> ‖<i>x</i><sub>i</sub> − μ<sub>'+(Number(r.cluster)+1)+'</sub>‖<sub>2</sub></span></div><div class="flow-focus"><b>Tâm cụm đang xét: μ'+(Number(r.cluster)+1)+'</b>Nhãn trên từng cạnh hiển thị khoảng cách Euclid thật trong vector 256D. PCA chỉ quyết định vị trí điểm trên hình, vì vậy độ dài cạnh không nhất thiết tỉ lệ với con số d.</div><section class="centroid-pipeline"><h3>Khoảng cách trong cụm và câu được chọn</h3>'+svg+rowTable(['Câu','Khoảng cách PCA 2D','Khoảng cách thật 256D','Trạng thái'],distanceRows)+'</section><div class="flow-result"><b>Đại diện thực tế: '+esc(r.selected_id)+'</b><br>d = '+num(r.selected_distance,9)+'. '+(outside?'Lưu ý: theo logic repository hiện tại, phép tìm gần tâm đang xét trên toàn bộ câu nên node được chọn nằm ngoài cụm; node vẫn được đánh dấu ★ nhưng không nối cạnh để giữ đúng yêu cầu chỉ nối các điểm thuộc cụm.':'Câu được chọn thuộc cụm đang xét và có khoảng cách nhỏ nhất theo phép chọn của pipeline.')+'</div>';
}
function centroidVectorStep(){
  const cent=DATA.centroid,meta={};(cent.process||[]).forEach(c=>(c.terms||[]).forEach(t=>{if(!meta[t.term])meta[t.term]=t;}));
  const allTerms=(cent.all_centroid_terms||[]).slice();
  const rows=allTerms.map(t=>{const m=meta[t.term]||{};return '<tr><td><b>'+esc(t.term)+'</b></td><td class="num">'+(m.df??'—')+'</td><td class="num">'+num(m.idf,7)+'</td><td class="num">'+num(t.raw_sum,7)+'</td><td class="num">'+num(t.normalized,7)+'</td><td class="num">'+num(t.after_threshold,7)+'</td><td>'+(Number(t.after_threshold)===0?'Bị ngưỡng loại':'Giữ trong vector')+'</td></tr>';});
  const top=allTerms.slice().sort((a,b)=>Number(b.after_threshold)-Number(a.after_threshold)).slice(0,18);
  const bars='<div class="mmr-bars">'+top.map(t=>{const v=Math.max(0,Math.min(1,Number(t.after_threshold)||0));return '<div class="mmr-bar-row '+(v>0?'winner':'')+'"><div class="mmr-bar-label">'+esc(t.term)+'</div><div class="mmr-bar-track"><div class="mmr-bar-fill" style="width:'+(v*100)+'%"></div></div><div class="mmr-bar-value">'+num(v,5)+'</div></div>';}).join('')+'</div>';
  return graphFlow([{title:'Tính TF-IDF',body:'Tạo trọng số TF-IDF cho từng token trong mỗi câu đại diện.'},{title:'Cộng theo token',body:'Cộng TF-IDF của token t qua toàn bộ câu đại diện.'},{title:'Chuẩn hóa',body:'Chia cho tổng trọng số lớn nhất trong vocabulary.'},{title:'Áp ngưỡng θtopic',body:'Nếu w(t) ≤ θtopic thì μ(t)=0; nếu không thì μ(t)=w(t).',kind:'primary-node'}])+
    '<div class="flow-equation math-equation"><span class="eq-line"><span class="math-op">TF-IDF</span>(<i>c</i>, <i>t</i>) = <span class="math-op">TF</span>(<i>t</i>, <i>c</i>) × <span class="math-op">IDF</span>(<i>t</i>)</span><span class="eq-line"><i>s</i>(<i>t</i>) = ∑<sub><i>c</i>∈R</sub> <span class="math-op">TF-IDF</span>(<i>c</i>, <i>t</i>)</span><span class="eq-line"><i>w</i>(<i>t</i>) = <span class="math-frac"><span><i>s</i>(<i>t</i>)</span><span>max<sub><i>u</i>∈V</sub> <i>s</i>(<i>u</i>)</span></span></span><span class="eq-line">μ(<i>t</i>) = <span class="math-piecewise"><span class="brace">{</span><span class="case">0, nếu <i>w</i>(<i>t</i>) ≤ θ<sub>topic</sub></span><span class="case"><i>w</i>(<i>t</i>), nếu <i>w</i>(<i>t</i>) &gt; θ<sub>topic</sub></span></span></span></div>'+
    '<p class="math-explain"><b>Ký hiệu:</b> R là tập câu đại diện; V là vocabulary; s(t) là tổng TF-IDF của token t; w(t) là trọng số sau chuẩn hóa; μ(t) là trọng số cuối cùng của vector chủ đề.</p>'+
    '<div class="metric-row">'+metrics([[(cent.all_centroid_terms||[]).length,'token trong vocabulary'],[num(cent.threshold_topic,3),'ngưỡng θtopic'],[num(cent.threshold,3),'ngưỡng τ lọc trùng'],[cent.selection_mode==='cosine'?'Cosine + ngưỡng':'CentroidBow gốc','chế độ chọn câu'],[num(cent.cosine_threshold,3),'ngưỡng cosine θ']])+'</div>'+
    '<section class="centroid-pipeline"><h3>Trọng số token của vector chủ đề μ</h3><p class="small">Thanh dài thể hiện trọng số sau chuẩn hóa và áp ngưỡng. Chỉ hiển thị tối đa 18 token có trọng số lớn nhất; bảng bên dưới liệt kê toàn bộ vocabulary.</p>'+bars+'</section>'+
    flowCaption('Bảng trọng số vector μ','DF/IDF được tính trên tập câu đại diện bằng CountVectorizer và TfidfTransformer')+
    rowTable(['Token t','DF(t)','IDF(t)','Tổng s(t)','Trọng số w(t)','Trọng số μ(t)','Trạng thái'],rows)+
    '<div class="flow-result"><b>Vector chủ đề μ đã sẵn sàng.</b><br>Bước tiếp theo dùng cùng vector μ để tính cosine của từng câu đại diện; chưa quyết định giữ/bỏ câu ở bước này.</div>';
}

function centroidOverviewStep(){
  const cent=DATA.centroid,items=(cent.process||[]).slice().sort((a,b)=>Number(a.index)-Number(b.index)),reps=DATA.representatives||[];
  const repRows=reps.map(r=>'<tr><td><b>'+esc(r.id)+'</b></td><td class="sentence">'+esc(r.text)+'</td><td>Cụm '+(r.cluster+1)+'</td><td>'+((r.members||[]).map(esc).join(', ')||'—')+'</td></tr>');
  const candidateRows=items.map((c,i)=>'<tr><td>'+(i+1)+'</td><td><b>'+esc(c.id)+'</b></td><td class="sentence">'+esc(c.text)+'</td><td>Chưa tính cosine</td></tr>');
  return '<div class="algorithm-note"><b>Mục tiêu bước 1/5:</b> xác định hai tập dữ liệu đầu vào. Các câu đại diện của K-means vừa là nguồn để dựng vector chủ đề, vừa là các ứng viên sẽ được chấm điểm. Ở bước này chưa tính TF-IDF, chưa tạo μ, chưa tính cosine và chưa loại câu.</div>'+
    graphFlow([{title:'K-means hoàn tất',body:DATA.kmeans.k+' cụm đã được tạo.'},{title:'Nhận câu đại diện',body:reps.length+' câu được chuyển từ bước K-means.'},{title:'Tạo tập ứng viên',body:items.length+' câu đại diện sẽ được chấm ở các bước sau.',kind:'primary-node'}])+
    '<div class="metric-row">'+metrics([[reps.length,'câu đại diện đầu vào'],[items.length,'ứng viên sẽ chấm'],['Chưa','tính TF-IDF ở bước này'],['Chưa','xếp hạng cosine ở bước này']])+'</div>'+
    '<section class="centroid-pipeline"><h3>Đầu vào A · Câu đại diện từ K-means</h3>'+rowTable(['Câu đại diện','Nội dung','Cụm','Các câu thuộc cụm'],repRows)+'</section>'+
    '<section class="centroid-pipeline"><h3>Đầu vào B · Các câu sẽ được tính cosine</h3><p class="small">Danh sách này giữ thứ tự đầu vào; chưa sắp xếp theo điểm và chưa có quyết định giữ/bỏ.</p>'+rowTable(['STT đầu vào','ID','Nội dung câu','Trạng thái hiện tại'],candidateRows)+'</section>'+
    '<div class="flow-equation math-equation"><span class="eq-line">R = {<i>r</i><sub>1</sub>, <i>r</i><sub>2</sub>, …, <i>r</i><sub>m</sub>}</span><span class="eq-line">X = {<i>x</i><sub>1</sub>, <i>x</i><sub>2</sub>, …, <i>x</i><sub>n</sub>}</span></div>'+
    '<p class="math-explain">Trong demo này, R là tập câu đại diện dùng để tính TF-IDF và xây dựng μ; X là tập ứng viên sẽ được so sánh với μ. Khi chuyển sang bước 2, hệ thống mới bắt đầu tính trọng số token.</p>';
}

function centroidRankingStep(){
  const cent=DATA.centroid,ranked=(cent.ranking||[]).slice().sort((a,b)=>(Number(b.cosine_score)-Number(a.cosine_score))||(Number(a.index)-Number(b.index)));
  const theta=Number(cent.cosine_threshold)||0;
  const bars=ranked.map((r,i)=>{
    const score=Math.max(0,Math.min(1,Number(r.cosine_score)||0));
    const pass=score>=theta;
    return '<div class="centroid-cosine-row"><div class="centroid-cosine-label">#'+(i+1)+' · '+esc(r.id)+'</div><div class="centroid-cosine-track"><div class="centroid-cosine-fill '+(pass?'passed':'failed')+'" style="width:'+(score*100)+'%"></div><div class="centroid-cosine-threshold" style="left:'+(Math.max(0,Math.min(1,theta))*100)+'%"></div></div><div class="centroid-cosine-value">'+num(score,7)+'</div></div>';
  }).join('');
  const rows=ranked.map((r,i)=>{
    const score=Number(r.cosine_score),pass=Number.isFinite(score)&&score>=theta;
    return '<tr><td><b>'+(i+1)+'</b></td><td><b>'+esc(r.id)+'</b></td><td class="sentence">'+esc(r.text)+'</td><td class="num"><b>'+num(r.cosine_score,9)+'</b></td><td class="num">'+num(theta,5)+'</td><td>'+(pass?'<span class="cosine-verdict pass">Đạt θ</span>':'<span class="cosine-verdict fail">Dưới θ</span>')+'</td><td>'+(pass?'Được đưa sang bước kiểm tra trùng':'Loại ở bước ngưỡng cosine')+'</td></tr>';
  });
  return graphFlow([{title:'Nhận điểm cosine',body:'Mỗi ứng viên đã có một điểm cos(xᵢ, μ).'},{title:'Sắp xếp giảm dần',body:'Điểm lớn hơn được xếp trước.',kind:'primary-node'},{title:'Xét từ hạng 1',body:'Duyệt theo thứ tự điểm đã sắp.'},{title:'Chưa lọc trùng',body:'Bộ lọc trùng lặp chạy ở bước kế tiếp.',kind:'good-node'}])+
    '<div class="flow-equation math-equation"><span class="eq-line">cos(<i>x</i><sub>i</sub>, μ) = <span class="math-frac"><span><i>x</i><sub>i</sub> · μ</span><span>‖<i>x</i><sub>i</sub>‖<sub>2</sub> ‖μ‖<sub>2</sub></span></span></span><span class="eq-line">cos(<i>x</i><sub>(1)</sub>, μ) ≥ cos(<i>x</i><sub>(2)</sub>, μ) ≥ ⋯ ≥ cos(<i>x</i><sub>(n)</sub>, μ)</span><span class="eq-line">Điều kiện qua ngưỡng: cos(<i>x</i><sub>i</sub>, μ) ≥ θ, với θ = '+num(theta,5)+'</span></div>'+
    '<p class="math-explain">Thứ hạng chỉ quyết định câu nào được xét trước. Điểm đạt θ chưa đồng nghĩa chắc chắn được giữ: câu còn phải qua bộ lọc trùng lặp và giới hạn số câu.</p>'+
    '<section class="centroid-pipeline"><h3>Thứ tự cosine</h3><p class="small">Mỗi thanh biểu thị cos(xᵢ, μ) trong khoảng 0–1; vạch dọc là ngưỡng θ. Màu xanh/cam chỉ cho biết đạt hay chưa đạt ngưỡng, không phải quyết định cuối.</p><div class="centroid-cosine-list">'+bars+'</div></section>'+
    '<section class="centroid-pipeline"><h3>Bảng xếp hạng</h3>'+rowTable(['Hạng','Câu','Nội dung','Cosine','Ngưỡng θ','Kiểm tra ngưỡng','Bước kế tiếp'],rows)+'</section>'+
    '<div class="flow-result"><b>Bước tiếp theo:</b> duyệt ứng viên theo hạng này. Câu dưới ngưỡng θ bị loại ngay; câu đạt ngưỡng mới được so với các câu đã giữ bằng similarity cặp câu.</div>';
}

function centroidCandidateScore(item){
  const cent=DATA.centroid,items=(cent.process||[]).slice().sort((a,b)=>Number(a.index)-Number(b.index));
  const rows=items.map((c,i)=>{
    const terms=(c.cosine_terms||[]);
    const detail=terms.map(t=>'<span class="eq-line"><i>'+esc(t.term)+'</i>: '+num(t.tfidf,4)+' × '+num(t.centroid_weight,4)+' = '+num(t.product,5)+'</span>').join('');
    const score=Number(c.cosine_score),pass=Number.isFinite(score)&&score>=Number(cent.cosine_threshold||0);
    return '<tr class="'+(pass?'won':'lost')+'"><td class="num">'+(i+1)+'</td><td><b>'+esc(c.id)+'</b></td><td class="sentence">'+esc(c.text)+'</td><td><div class="math-equation">'+detail+'</div></td><td class="num">'+num(c.cosine_dot,8)+'</td><td class="num">'+num(c.cosine_sentence_norm,8)+'</td><td class="num">'+num(c.cosine_centroid_norm,8)+'</td><td class="num"><b>'+num(c.cosine_score,9)+'</b></td><td>'+(pass?'Đạt θ':'Dưới θ')+'</td></tr>';
  });
  return flowBadge('Bước 3/5 · Tính cosine cho từng ứng viên')+
    graphFlow([{title:'Vector câu xᵢ',body:'TF-IDF của từng token trong ứng viên.'},{title:'Tích vô hướng',body:'Cộng TF-IDF(cᵢ,t) × μ(t).'},{title:'Chuẩn vector',body:'Tính ‖xᵢ‖₂ và ‖μ‖₂.'},{title:'Cosine',body:'Chia tích vô hướng cho tích hai chuẩn.',kind:'primary-node'}])+
    '<div class="flow-equation math-equation"><span class="eq-line"><i>x</i><sub>i</sub> · μ = ∑<sub><i>t</i>∈V</sub> <span class="math-op">TF-IDF</span>(<i>c</i><sub>i</sub>, <i>t</i>) μ(<i>t</i>)</span><span class="eq-line">cos(<i>x</i><sub>i</sub>, μ) = <span class="math-frac"><span><i>x</i><sub>i</sub> · μ</span><span>‖<i>x</i><sub>i</sub>‖<sub>2</sub> ‖μ‖<sub>2</sub></span></span></span><span class="eq-line">θ = '+num(cent.cosine_threshold,5)+'</span></div>'+
    '<p class="math-explain">Mỗi dòng trong bảng là một câu. Cột “Đóng góp theo token” liệt kê tối đa 20 tích dương lớn nhất để bảng dễ đọc; cột tử số vẫn là tổng trên toàn bộ vocabulary. Hai cột chuẩn vector tạo thành mẫu số. Cosine được tính cho toàn bộ ứng viên trước khi bắt đầu xếp hạng.</p>'+
    '<div class="flow-result"><b>Đã tính cosine cho '+items.length+' ứng viên.</b> Bảng dưới đây giữ thứ tự đầu vào, chưa sắp xếp theo điểm. Bước kế tiếp mới xếp cosine giảm dần.</div>'+
    '<section class="centroid-pipeline centroid-cosine-table"><h3>Bảng tính cosine đầy đủ · tất cả ứng viên</h3>'+rowTable(['# đầu vào','Câu','Nội dung','Đóng góp theo token: TF-IDF × μ(t)','Tử số xᵢ · μ','‖xᵢ‖₂','‖μ‖₂','cos(xᵢ, μ)','Kiểm tra θ'],rows)+'</section>';
}

function centroidCompareStep(item){const cent=DATA.centroid,c=cent.process[item.centroidIndex],p=c?.comparisons?.[item.comparisonIndex];if(!c||!p)return '';const rejected=Number(p.similarity)>Number(cent.threshold);return '<div class="flow-focus"><b>Ứng viên đang xét: '+esc(c.id)+'</b>'+esc(c.text)+'</div><div class="flow-equation">Sim('+esc(c.id)+', '+esc(p.against_id)+') = '+num(p.similarity,9)+'<br>Ngưỡng lọc trùng = '+num(p.threshold,5)+'</div><section class="mmr-branch '+(rejected?'red':'score')+'"><h4>Kết quả so sánh</h4><p>'+esc(p.against_id)+' · '+esc(p.against_text)+'</p><div class="flow-equation">'+num(p.similarity,9)+' '+(rejected?'>':'≤')+' '+num(p.threshold,5)+'</div><div class="flow-result '+(rejected?'':'good')+'"><b>'+(rejected?'Vượt ngưỡng → loại ứng viên':'Không vượt ngưỡng → phép so sánh này cho phép giữ')+'</b>'+(!rejected?' Tuy nhiên, ứng viên còn phải qua các phép so sánh với các câu đã giữ khác.':'')+'</div></section><details class="flow-hidden-detail"><summary>Chi tiết similarity nếu có</summary>'+esc(JSON.stringify(p,null,2))+'</details>';}
function centroidDecisionStep(item){
  const items=DATA.centroid.process||[],c=items[item.centroidIndex];if(!c)return '';
  const cent=DATA.centroid,theta=Number(cent.cosine_threshold)||0,tau=Number(cent.threshold)||0;
  const ranked=(cent.ranking||[]).slice().sort((a,b)=>(Number(b.cosine_score)-Number(a.cosine_score))||(Number(a.index)-Number(b.index)));
  const rankPosition=ranked.findIndex(r=>Number(r.index)===Number(c.index));
  const rankNo=rankPosition+1,cosineValue=Number(c.cosine_score),passes=Number.isFinite(cosineValue)&&cosineValue>=theta;
  const atLimit=!c.selected&&/đủ giới hạn 20 câu/i.test(String(c.reason||''));
  const comparisonRows=(c.comparisons||[]).map(p=>{
    const threshold=Number(p.threshold??tau),rejected=Number(p.similarity)>threshold;
    return '<tr class="'+(rejected?'lost':'won')+'"><td><b>'+esc(p.against_id)+'</b></td><td class="sentence">'+esc(p.against_text)+'</td><td class="num">'+num(p.similarity,8)+'</td><td class="num">'+num(threshold,5)+'</td><td>'+(rejected?'<span class="cosine-verdict fail">Vượt τ · loại</span>':'<span class="cosine-verdict pass">Không vượt τ</span>')+'</td></tr>';
  });
  let reason=c.reason||'';
  if(atLimit) reason='Bị loại vì đã đạt giới hạn 20 câu được chọn.';
  else if(!passes) reason='Bị loại vì cos(xᵢ, μ) = '+num(c.cosine_score,7)+' < θ = '+num(theta,5)+'.';
  else if(c.selected) reason='Cosine đạt ngưỡng θ và không có similarity cặp câu nào vượt ngưỡng τ.';
  else if((c.comparisons||[]).some(p=>Number(p.similarity)>Number(p.threshold??tau))) reason='Cosine đạt θ nhưng bị loại vì similarity với một câu đã giữ vượt τ.';
  const decisionTitle=c.selected?'✓ GIỮ '+esc(c.id):'✕ LOẠI '+esc(c.id);
  const decisionFlow=graphFlow([
    {title:'1 · Kiểm tra cosine',body:'cos = '+num(c.cosine_score,5)+'; θ = '+num(theta,5),kind:passes?'good-node':'warn-node'},
    {title:passes?'2 · Kiểm tra trùng':'2 · Bỏ qua kiểm tra trùng',body:!passes?'Cosine dưới ngưỡng nên dừng tại đây.':atLimit?'Đã đạt giới hạn 20 câu.':(c.comparisons||[]).length+' phép so với câu đã giữ.'},
    {title:'3 · Quyết định',body:c.selected?'Giữ và chuyển sang MMR.':'Loại, không chuyển sang MMR.',kind:c.selected?'good-node':'warn-node'}
  ]);
  return decisionFlow+'<div class="flow-focus '+(c.selected?'good':'warn')+'"><b>Hạng '+rankNo+' / '+ranked.length+' · '+decisionTitle+'</b>'+esc(reason)+'</div>'+
    '<div class="flow-equation math-equation"><span class="eq-line">Ứng viên: <i>c</i><sub>i</sub> = '+esc(c.id)+'</span><span class="eq-line">cos(<i>x</i><sub>i</sub>, μ) = '+num(c.cosine_score,9)+'</span><span class="eq-line">θ = '+num(theta,5)+' &nbsp; ⇒ &nbsp; '+num(c.cosine_score,9)+' '+(passes?'≥':'<')+' '+num(theta,5)+'</span></div>'+
    '<section class="centroid-pipeline"><h3>1 · Điều kiện cosine</h3><div class="cosine-meter"><div class="cosine-meter-fill" style="width:'+Math.max(0,Math.min(100,cosineValue*100||0))+'%"></div></div><div class="cosine-threshold-row"><div class="cosine-threshold-mark" style="left:'+Math.max(0,Math.min(100,theta*100))+'%"></div></div><p class="small">Chỉ khi cos(xᵢ, μ) ≥ θ, ứng viên mới được xét tiếp. Nếu thấp hơn θ, câu bị loại ngay.</p></section>'+
    '<section class="centroid-pipeline"><h3>2 · Bộ lọc trùng lặp</h3><div class="flow-equation math-equation"><span class="eq-line">Nếu cos(<i>x</i><sub>i</sub>, μ) &lt; θ ⇒ loại.</span><span class="eq-line">Nếu cos(<i>x</i><sub>i</sub>, μ) ≥ θ: so với từng câu đã giữ <i>s</i><sub>j</sub>.</span><span class="eq-line"><span class="math-op">sim</span>(<i>c</i><sub>i</sub>, <i>s</i><sub>j</sub>) &gt; τ ⇒ loại vì trùng lặp, với τ = '+num(tau,5)+'</span><span class="eq-line">Nếu không vượt τ với các câu đã giữ ⇒ giữ ứng viên.</span></div><p class="math-explain">Trong triển khai này, hàm lọc trùng nhận ứng viên ở đối số thứ nhất và câu đã giữ ở đối số thứ hai: sim(cᵢ, sⱼ). Không đảo thứ tự hai đối số khi diễn giải kết quả.</p>'+
    (!passes?'<div class="flow-result"><b>Không thực hiện so trùng.</b> Câu không đạt ngưỡng cosine nên bị loại ở điều kiện 1.</div>':atLimit?'<div class="flow-result"><b>Đã chạm giới hạn số câu.</b> Ứng viên không được thêm vào danh sách dù cosine đạt ngưỡng.</div>':comparisonRows.length?rowTable(['Câu đã giữ sⱼ','Nội dung','sim(cᵢ, sⱼ)','Ngưỡng τ','Kết quả'],comparisonRows):'<div class="flow-result">Chưa có câu đã giữ cần so sánh; ứng viên qua điều kiện chống trùng nếu không vướng giới hạn 20 câu.</div>')+'</section>'+
    '<div class="flow-result"><b>Quyết định cho '+esc(c.id)+': '+(c.selected?'GIỮ → chuyển sang MMR':'LOẠI → không chuyển sang MMR')+'</b><br>'+esc(reason)+'</div>'+
    flowCaption('Các câu đã xét theo thứ hạng cosine đến thời điểm này','Trạng thái được cập nhật theo đúng thứ tự duyệt')+rowTable(['Hạng','Câu','Cosine','Điều kiện θ','Trạng thái'],ranked.slice(0,rankPosition+1).map((r,i)=>{const p=items.find(x=>Number(x.index)===Number(r.index));if(!p)return '';const active=Number(r.index)===Number(c.index),score=Number(p.cosine_score),pass=Number.isFinite(score)&&score>=theta;const outcome=active?(p.selected?'ĐANG GIỮ':'ĐANG LOẠI'):(p.selected?'Đã giữ':'Đã loại');return '<tr class="'+(active?'won':'')+'"><td>'+(i+1)+'</td><td><b>'+esc(p.id)+'</b></td><td class="num">'+num(p.cosine_score,8)+'</td><td>'+(pass?'Đạt':'Dưới θ')+'</td><td>'+(p.selected?'<span class="pill">'+outcome+'</span>':outcome)+'</td></tr>';}));
}

function mmrQueryStep(){
  const q=DATA.mmr.pool[0],chosen=[q],candidates=DATA.mmr.pool.slice(1);
  return '<div class="flow-focus good"><b>BƯỚC 1 · KHỞI TẠO TẬP CÂU ĐÃ CHỌN R</b>Q được khởi tạo theo quy tắc của pipeline: lấy câu đầu tiên sau Centroid làm câu mốc, sau đó đưa câu này vào R.</div>'+graphFlow([{title:'Đầu vào',body:'Các câu sau bước CentroidBow.',kind:'primary-node'},{title:'Chọn câu khởi tạo Q',body:q.id+' · quy tắc khởi tạo hiện tại của pipeline.'},{title:'Khởi tạo tập R',body:'R = { '+q.id+' }',kind:'good-node'},{title:'Danh sách chưa chọn U',body:candidates.length?candidates.map(x=>x.id).join(' · '):'∅'}])+'<div class="flow-equation">Q = '+esc(q.id)+'<br>R₀ = {Q}<br>U₀ = Pool sau Centroid trừ {Q}<br>Q cố định qua mọi vòng; sau mỗi vòng, câu thắng được thêm vào R và bỏ khỏi U.</div><div class="mmr-network"><div class="mmr-network-grid"><div class="mmr-network-node qnode"><b>Q · cố định</b><p>'+esc(q.id)+' — '+esc(q.text)+'</p></div><div class="mmr-network-node snode"><b>R₀ · tập đã chọn</b><p>{ '+esc(q.id)+' }</p></div><div class="mmr-network-node cnode"><b>U₀ · ứng viên chưa chọn</b><p>'+candidates.map(x=>esc(x.id)).join(' · ')+'</p></div></div><div class="graph-caption">Nếu R bắt đầu rỗng, cần dùng quy tắc khởi tạo của hệ thống để chọn câu đầu tiên. Pipeline này khởi tạo bằng câu đầu sau CentroidBow.</div></div><div class="flow-table-caption"><h3>Trạng thái ban đầu</h3><span>Q giữ nguyên qua tất cả lượt</span></div>'+rowTable(['Vai trò','Câu'],['<tr><td>Truy vấn Q</td><td><b>'+esc(q.id)+'</b> · '+esc(q.text)+'</td></tr>','<tr><td>Tập đã chọn R</td><td>'+chips([q.id+' · '+q.text])+'</td></tr>','<tr><td>Ứng viên chưa chọn U</td><td>'+(candidates.length?chips(candidates.map(x=>x.id+' · '+x.text)):'∅')+'</td></tr>']);
}
function mmrPhaseState(round){
  const q=DATA.mmr.pool[0],before=round.selected_before||[],remaining=round.remaining_before||[];
  return '<div class="mmr-state-row"><div class="mmr-state-card q"><b>Q · truy vấn cố định</b><p>'+esc(q?.id||'Q')+' · '+esc(q?.text||'')+'</p><p><strong>Không thay đổi giữa các vòng.</strong></p></div><div class="mmr-state-card s"><b>R đã chọn trước vòng · '+before.length+' câu</b><p>'+(before.map(x=>esc(x.id)+' · '+esc(x.text)).join('<br>')||'∅')+'</p></div><div class="mmr-state-card r"><b>Ứng viên U trước vòng · '+remaining.length+' câu</b><p>'+(remaining.map(x=>esc(x.id)).join(' · ')||'∅')+'</p></div></div>';
}
function mmrMetricBars(candidates,key){
  const vals=candidates.map(c=>Number(c[key])).filter(Number.isFinite),lo=vals.length?Math.min(...vals):0,hi=vals.length?Math.max(...vals):1,span=hi-lo||1;
  return '<div class="mmr-bars">'+candidates.slice().sort((a,b)=>Number(b[key])-Number(a[key])).map(c=>{const v=Number(c[key]),width=Math.max(4,12+88*(v-lo)/span);return '<div class="mmr-bar-row '+(c.winner?'winner':'')+'"><div class="mmr-bar-label">'+esc(c.id)+(c.winner?' ★':'')+'</div><div class="mmr-bar-track"><div class="mmr-bar-fill '+(key==='red'?'red-fill':'')+'" style="width:'+width+'%"></div></div><div class="mmr-bar-value">'+num(v,8)+'</div></div>';}).join('')+'</div>';
}
function mmrRelOverviewStep(item){
  const r=DATA.mmr.rounds[item.roundIndex],cs=r.candidates||[];
  const rows=cs.map(c=>'<tr><td><b>'+esc(c.id)+'</b></td><td class="sentence">'+esc(c.text)+'</td><td class="num"><b>'+num(c.rel,9)+'</b></td><td>sentenceSim(c, Q)</td></tr>');
  return '<div class="flow-focus"><b>BƯỚC 2 · ĐÁNH GIÁ ỨNG VIÊN — ĐỘ LIÊN QUAN</b>Vòng '+(item.roundIndex+1)+': duyệt từng câu chưa chọn và tính độ liên quan với Q.</div>'+mmrPhaseState(r)+graphFlow([{title:'Giữ Q cố định',body:DATA.mmr.pool[0].id+' không đổi.'},{title:'Duyệt từng ứng viên U',body:'Ứng viên chưa chọn được xét lần lượt.'},{title:'Tính Rel(c)',body:'Rel(c) = sentenceSim(c, Q).',kind:'primary-node'},{title:'Lưu điểm liên quan',body:'Dùng Rel ở công thức MMR.'}])+flowCaption('Điểm liên quan của ứng viên chưa chọn','Phần này mới hiển thị Rel')+mmrMetricBars(cs,'rel')+rowTable(['Ứng viên c','Nội dung','Rel(c)','Cách tính'],rows)+'<div class="flow-equation">Rel(c) = sentenceSim(c, Q)<br>Q = '+esc(DATA.mmr.pool[0].id)+' · '+esc(DATA.mmr.pool[0].text)+'</div><details class="flow-hidden-detail"><summary>Mở phép tính sentenceSim() theo từng token</summary>'+cs.map(c=>'<details class="flow-hidden-detail"><summary>'+esc(c.id)+' · Rel = '+num(c.rel,8)+'</summary>'+simDetails(c.rel_detail)+'</details>').join('')+'</details>';
}
function mmrRedOverviewStep(item){
  const r=DATA.mmr.rounds[item.roundIndex],cs=r.candidates||[],before=r.selected_before||[];
  const headers=['Ứng viên c',...before.map(s=>'Sim(c, '+s.id+')'),'Red(c,R) = max'];
  const rows=cs.map(c=>{const pairs=c.red_details||[];const vals=before.map(s=>{const p=pairs.find(x=>x.against_id===s.id);return p?'<td class="num '+(p.is_max?'max-red':'')+'">'+num(p.score,8)+(p.is_max?' ★':'')+'</td>':'<td class="num">—</td>';}).join('');return '<tr><td><b>'+esc(c.id)+'</b></td>'+vals+'<td class="num"><b>'+num(c.red,9)+'</b></td></tr>';});
  const details=cs.map(c=>'<tr><td><b>'+esc(c.id)+'</b></td><td>'+((c.red_details||[]).map(p=>esc(p.against_id)+' = '+num(p.score,7)+(p.is_max?' ★':'')).join('<br>')||'Không có câu trong R để so')+'</td><td class="num">'+num(c.red,9)+'</td></tr>');
  return '<div class="flow-focus"><b>BƯỚC 2 · ĐÁNH GIÁ ỨNG VIÊN — ĐỘ TRÙNG LẶP</b>Với mỗi ứng viên, so độ tương đồng với từng câu đang nằm trong R; điểm lớn nhất là Red(c,R).</div>'+mmrPhaseState(r)+graphFlow([{title:'Lấy tập R hiện tại',body:'R gồm Q và các câu thắng từ những vòng trước.'},{title:'So từng cặp (c,r)',body:'Với mỗi ứng viên c trong U, tính Sim(c,r) cho mọi r ∈ R.'},{title:'Lấy giá trị lớn nhất',body:'Red(c,R) = maxᵣ Sim(c,r).',kind:'primary-node'},{title:'Chuẩn bị điểm phạt',body:'Red càng cao thì điểm phạt trùng lặp càng lớn.',kind:'warn-node'}])+'<div class="flow-equation">Red(c,R) = max<sub>r∈R</sub> sentenceSim(c,r)</div>'+flowCaption('Ma trận so sánh ứng viên với các câu đã chọn','★ đánh dấu similarity lớn nhất của mỗi ứng viên')+rowTable(headers,rows)+flowCaption('Điểm Red của mọi ứng viên','Thanh cam dài hơn = mức trùng lặp / điểm phạt cao hơn')+'<div class="mmr-red-legend"><b>Đọc đúng biểu đồ:</b> Red càng lớn thì câu càng giống ít nhất một câu đã nằm trong R, nên phần phạt trong công thức MMR càng lớn. Red cao không có nghĩa câu đó tốt hơn.</div>'+mmrMetricBars(cs,'red')+flowCaption('Cách lấy max cho từng ứng viên')+rowTable(['Ứng viên','Các phép so sánh với R','Red sau khi lấy max'],details)+'<details class="flow-hidden-detail"><summary>Mở phép tính sentenceSim() cho các cặp ứng viên–câu trong R</summary>'+cs.map(c=>'<details class="flow-hidden-detail"><summary>'+esc(c.id)+' · Red = '+num(c.red,8)+'</summary>'+(c.red_details||[]).map(p=>'<details class="flow-hidden-detail"><summary>Sim('+esc(c.id)+', '+esc(p.against_id)+') = '+num(p.score,8)+(p.is_max?' ★ max':'')+'</summary>'+simDetails(p)+'</details>').join('')+'</details>').join('')+'</details>';
}
function mmrScoreOverviewStep(item){
  const r=DATA.mmr.rounds[item.roundIndex],lambda=Number(DATA.mmr.lambda),cs=r.candidates||[];
  return '<div class="flow-focus"><b>BƯỚC 3 · TÍNH ĐIỂM MMR</b>Kết hợp Rel(c) với điểm phạt trùng lặp Red(c,R). Câu liên quan hơn và ít giống các câu đã chọn thường có điểm cao hơn.</div>'+mmrPhaseState(r)+graphFlow([{title:'Điểm liên quan',body:'Rel(c) = sentenceSim(c,Q).'},{title:'Điểm trùng lặp',body:'Red(c,R) = max similarity với tập R.'},{title:'Thay số vào công thức',body:'MMR(c) = λ × Rel(c) − (1−λ) × Red(c,R).',kind:'primary-node'},{title:'So sánh sau khi chấm',body:'Chỉ công bố câu thắng sau khi mọi ứng viên đã được tính điểm.',kind:'good-node'}])+'<div class="flow-equation">MMR(c) = λ × Rel(c) − (1−λ) × Red(c,R)<br>λ = '+num(lambda,4)+'; 1−λ = '+num(1-lambda,4)+'</div><div class="mmr-flow"><section class="mmr-branch rel"><h4>Thành phần thưởng · Rel</h4><p>Độ liên quan với Q càng cao thì điểm được tăng càng nhiều.</p></section><section class="mmr-branch red"><h4>Thành phần phạt · Red</h4><p>Độ giống với các câu trong R càng cao thì bị trừ càng nhiều.</p></section><section class="mmr-branch score"><h4>Quy tắc chọn</h4><p>Tính điểm cho '+cs.length+' ứng viên rồi lấy điểm lớn nhất.</p></section></div><div class="flow-focus"><b>Vòng '+(item.roundIndex+1)+' có '+cs.length+' ứng viên.</b> Các màn hình kế tiếp sẽ thay số cho từng câu riêng lẻ. Chưa câu nào được chuyển vào R cho đến khi xét hết ứng viên.</div>';
}
function mmrRelStep(item){const r=DATA.mmr.rounds[item.roundIndex],c=r.candidates[item.candidateIndex];return '<div class="flow-focus"><b>Ứng viên '+esc(c.id)+'</b>'+esc(c.text)+'</div><div class="flow-equation">Rel(c) = sentenceSim(c, Q) = '+num(c.rel,9)+'<br>Q = '+esc(DATA.mmr.pool[0].id)+' · '+esc(DATA.mmr.pool[0].text)+'</div>'+simDetails(c.rel_detail);}
function mmrRedPairStep(item){const r=DATA.mmr.rounds[item.roundIndex],c=r.candidates[item.candidateIndex],p=c.red_details[item.pairIndex];return '<div class="flow-focus"><b>Tính một thành phần Red cho '+esc(c.id)+'</b>So ứng viên với câu '+esc(p.against_id)+' đang nằm trong tập R.</div><div class="flow-equation">Sim('+esc(c.id)+', '+esc(p.against_id)+') = '+num(p.score,9)+(p.is_max?' ← giá trị lớn nhất hiện tại':'')+'</div>'+simDetails(p);}
function mmrRedMaxStep(item){const r=DATA.mmr.rounds[item.roundIndex],c=r.candidates[item.candidateIndex],pairs=c.red_details||[];const rows=pairs.map(p=>'<tr class="'+(p.is_max?'max-red':'')+'"><td>'+esc(p.against_id)+'</td><td class="num">'+num(p.score,9)+'</td><td>'+(p.is_max?'<span class="pill">Dùng làm Red</span>':'Không phải max')+'</td></tr>');return '<div class="flow-equation">Red(c,R) = max<sub>r∈R</sub> Sim(c,r)</div>'+rowTable(['Câu r ∈ R','Similarity','Kết quả'],rows)+'<div class="flow-result"><b>Red('+esc(c.id)+', R) = '+num(c.red,9)+'</b><br>Giá trị lớn nhất trong bảng được dùng làm điểm phạt trùng lặp.</div>';}
function mmrScoreStep(item){
  const r=DATA.mmr.rounds[item.roundIndex],c=r.candidates[item.candidateIndex],lambda=Number(DATA.mmr.lambda),formula=c.formula_score,match=c.score_matches_formula;
  // Reveal only the candidate currently under inspection. The argmax is shown after all candidates have been scored.
  const currentRow='<tr><td><b>'+esc(c.id)+'</b><br><span class="small">'+esc(c.text)+'</span></td><td class="num">'+num(c.rel,9)+'</td><td class="num">'+num(c.red,9)+'</td><td class="num"><b>'+num(formula,10)+'</b></td><td class="num"><b>'+num(c.score,10)+'</b></td></tr>';
  return '<div class="flow-focus"><b>Vòng '+(item.roundIndex+1)+' · Đang tính cho '+esc(c.id)+'</b>'+esc(c.text)+'</div>'+ 
    '<div class="mmr-flow"><section class="mmr-branch rel"><h4>1 · Rel(c)</h4><div class="flow-equation">Rel = '+num(c.rel,9)+'</div><p>Độ liên quan của '+esc(c.id)+' với Q.</p></section><section class="mmr-branch red"><h4>2 · Red(c,R)</h4><div class="flow-equation">Red = '+num(c.red,9)+'</div><p>Độ giống lớn nhất với các câu đang nằm trong R.</p></section><section class="mmr-branch score"><h4>3 · Thay số vào MMR</h4><div class="flow-equation">λ = '+num(lambda,4)+'<br>MMR(c) = λ×Rel − (1−λ)×Red<br>= '+num(lambda,4)+'×'+num(c.rel,9)+' − '+num(1-lambda,4)+'×'+num(c.red,9)+'<br>= <b>'+num(formula,10)+'</b></div><p>MMRScore() thực tế = <b>'+num(c.score,10)+'</b></p></section></div>'+ 
    '<div class="flow-equation">MMR('+esc(c.id)+') = '+num(lambda,5)+' × '+num(c.rel,9)+' − '+num(1-lambda,5)+' × '+num(c.red,9)+' = <b>'+num(formula,10)+'</b></div>'+ 
    (match?'<div class="flow-result">✓ Điểm tính theo công thức khớp với MMRScore() của repository.</div>':'<div class="note warn">Công thức tường minh = '+num(formula,10)+'; MMRScore() thực tế = '+num(c.score,10)+'. Kết quả so sánh cuối vòng sẽ dùng MMRScore() thật của repository.</div>')+ 
    '<div class="flow-focus"><b>Đã tính xong '+esc(c.id)+'; chưa công bố câu thắng.</b> Xét tiếp ứng viên kế tiếp theo đúng công thức. Khi tất cả ứng viên đã được chấm, bước sau mới so điểm và chuyển câu có điểm cao nhất từ U sang R.</div>'+ 
    flowCaption('Bảng tính riêng cho câu đang xét','Các ứng viên khác sẽ xuất hiện sau khi được duyệt')+rowTable(['Ứng viên','Rel(c)','Red(c,R)','Theo công thức','MMRScore() thật'],[currentRow]);
}
function mmrWinnerStep(item){
  const r=DATA.mmr.rounds[item.roundIndex],rows=r.candidates.map(c=>'<tr class="'+(c.winner?'won':'')+'"><td><b>'+esc(c.id)+'</b><br><span class="small">'+esc(c.text)+'</span></td><td class="num">'+num(c.rel,7)+'</td><td class="num">'+num(c.red,7)+'</td><td class="num"><b>'+num(c.formula_score,9)+'</b></td><td class="num"><b>'+num(c.score,9)+'</b></td><td>'+(c.winner?'<span class="pill">★ Thắng — chuyển vào R</span>':'Giữ trong danh sách ứng viên')+'</td></tr>');
  return '<div class="flow-focus good"><b>BƯỚC 4 · CHỌN CÂU CÓ ĐIỂM MMR CAO NHẤT</b>Đã chấm toàn bộ '+r.candidates.length+' ứng viên. Bây giờ chọn '+esc(r.winner_id)+' và chuyển câu này từ danh sách chưa chọn vào R.</div>'+mmrTransferBoard(r)+flowCaption('Điểm MMR của mọi ứng viên ở vòng '+(item.roundIndex+1),'Câu thắng được tô xanh')+mmrScoreBars(r.candidates)+flowCaption('So sánh điểm và quyết định')+rowTable(['Ứng viên','Rel','Red','Theo công thức','MMRScore() thật','Quyết định'],rows)+'<div class="flow-result"><b>✓ Đã thêm '+esc(r.winner_id)+' vào R.</b><br>Ứng viên này được bỏ khỏi U. Ở vòng tiếp theo, Red của từng câu còn lại sẽ được tính lại với R mới.</div>';
}
function mmrLoopCheckStep(item){
  if(item.roundIndex===-1){
    const q=DATA.mmr.pool?.[0],target=DATA.mmr.target;
    return '<div class="flow-focus warn"><b>BƯỚC 5 · KIỂM TRA ĐIỀU KIỆN DỪNG</b>Không có vòng chọn bổ sung trong lần chạy này. Pipeline giữ câu khởi tạo Q trong R theo quy tắc hiện tại; không có lượt bổ sung nào được thực hiện.</div>'+graphFlow([{title:'R hiện tại',body:q?'R = { '+q.id+' }':'R = ∅',kind:'good-node'},{title:'Kiểm tra U',body:DATA.mmr.pool.length>1?'Còn câu ứng viên nhưng không có vòng trong cấu hình chạy này.':'Không còn ứng viên chưa chọn.'},{title:'Kết thúc',body:'target theo chế độ = '+String(target??'—')+'.',kind:'warn-node'}])+'<div class="flow-equation">Q được chọn làm câu khởi tạo theo quy tắc của repository.<br>Không có lượt MMR bổ sung được thực hiện.</div>';
  }
  const rounds=DATA.mmr.rounds||[],r=rounds[item.roundIndex],remaining=r?.remaining_after||[],selected=r?.selected_after||[],hasNext=item.roundIndex<rounds.length-1&&remaining.length>0;
  const target=DATA.mmr.target;
  const status=hasNext?'TIẾP TỤC':'DỪNG';
  const explain=hasNext?'Vẫn còn ứng viên và lần chạy còn vòng tiếp theo. Sang vòng kế, tính lại Rel/Red/MMR; Red thay đổi theo tập R mới.':(remaining.length===0?'Danh sách ứng viên đã hết.':'Đã hết số vòng theo giới hạn của chế độ chạy hiện tại.');
  const before=r?.selected_before||[],winner=(r?.candidates||[]).find(c=>c.winner),winnerText=winner?winner.id+' · '+winner.text:(r?.winner_id||'—');
  return '<div class="flow-focus '+(hasNext?'good':'warn')+'"><b>BƯỚC 5 · '+status+' — CẬP NHẬT R VÀ KIỂM TRA ĐIỀU KIỆN DỪNG</b>'+explain+'</div>'+    '<div class="mmr-loop-state"><div class="state-r"><b>R trước lượt</b><p>'+(before.map(x=>esc(x.id)).join(' · ')||'∅')+'</p></div><div class="state-w"><b>Câu vừa thắng</b><p>'+esc(winnerText)+'</p></div><div class="state-r"><b>R sau lượt</b><p>'+(selected.map(x=>esc(x.id)).join(' · ')||'∅')+'</p></div></div>'+    graphFlow([{title:'Tập R đã cập nhật',body:'R = { '+selected.map(x=>x.id).join(', ')+' }',kind:'good-node'},{title:'Kiểm tra ứng viên U',body:remaining.length?remaining.map(x=>x.id).join(' · '):'U = ∅'},{title:'Kiểm tra giới hạn',body:'target theo chế độ = '+String(target??'—')+'.'},{title:hasNext?'Quay lại Bước 2':'Kết thúc MMR',body:hasNext?'Chấm lại toàn bộ ứng viên còn lại theo R mới.':'Không có vòng tiếp theo trong lần chạy này.',kind:hasNext?'primary-node':'warn-node'}])+'<div class="mmr-transfer-equation">R sau vòng '+(item.roundIndex+1)+' = { '+selected.map(x=>esc(x.id)).join(', ')+' }<br>Ứng viên còn lại U = { '+remaining.map(x=>esc(x.id)).join(', ')+' }</div><div class="flow-result"><b>'+status+': '+(hasNext?'tiếp tục vòng '+(item.roundIndex+2):'đạt điều kiện dừng')+'.</b><br>Q vẫn cố định nên Rel(c) so với Q thường giữ nguyên; R đổi khiến Red(c,R) phải tính lại, vì vậy điểm MMR của ứng viên chưa chọn có thể tăng hoặc giảm.</div>';
}
function mmrLiveCalc(item,c,r){
  if(!c||!r)return '';
  const lambda=Number(DATA.mmr.lambda),oneMinus=1-lambda;
  if(item.type==='mmr-candidate-rel'){
    return '<div class="mmr-live-calc rel-calc"><b>BƯỚC 2 · ĐỘ LIÊN QUAN</b><div class="mmr-live-equation">Rel('+esc(c.id)+') = Sim('+esc(c.id)+', Q)<br>= <strong>'+num(c.rel,8)+'</strong></div><p>Q cố định: '+esc(DATA.mmr.pool[0]?.id||'Q')+'. Giá trị này dùng làm thành phần thưởng liên quan.</p></div>';
  }
  if(item.type==='mmr-candidate-red'){
    const pairs=c.red_details||[];
    const pairHtml=pairs.length?pairs.map(p=>'<div class="mmr-live-pair '+(p.is_max?'max':'')+'"><span>Sim('+esc(c.id)+', '+esc(p.against_id)+')</span><b>'+num(p.score,8)+'</b>'+(p.is_max?'<small>MAX → Red</small>':'')+'</div>').join(''):'<p>R chưa có câu để so sánh.</p>';
    return '<div class="mmr-live-calc red-calc"><b>BƯỚC 2 · ĐỘ TRÙNG LẶP</b>'+pairHtml+'<div class="mmr-live-equation">Red('+esc(c.id)+', R) = max similarity<br>= <strong>'+num(c.red,8)+'</strong></div><p>Red càng lớn thì mức phạt trùng lặp càng cao.</p></div>';
  }
  if(item.type==='mmr-candidate-score'){
    const match=!!c.score_matches_formula;
    return '<div class="mmr-live-calc score-calc"><b>BƯỚC 3 · ĐIỂM MMR</b><div class="mmr-live-equation">MMR = λ × Rel − (1−λ) × Red<br>= '+num(lambda,3)+' × '+num(c.rel,8)+' − '+num(oneMinus,3)+' × '+num(c.red,8)+'<br>= <strong>'+num(c.formula_score,9)+'</strong></div><div class="mmr-live-score">MMRScore() thực tế = <b>'+num(c.score,9)+'</b></div>'+(match?'<p>✓ Điểm khớp với công thức.</p>':'<p class="score-warn">Lựa chọn cuối dùng MMRScore() thực tế từ repository.</p>')+'</div>';
  }
  if(item.type==='mmr-winner'){
    return '<div class="mmr-live-calc winner-calc"><b>BƯỚC 4 · CÂU ĐƯỢC CHỌN</b><div class="mmr-live-equation">argmax MMRScore(c)<br>= <strong>'+esc(r.winner_id)+'</strong><br>Điểm = '+num(c.score,9)+'</div><p>Câu thắng được chuyển sang R; các câu còn lại tiếp tục ở U.</p></div>';
  }
  return '';
}
function mmrLiveCard(x,opts={}){
  const active=!!opts.active,winner=!!opts.winner,done=!!opts.done;
  let cls='mmr-live-card';if(active)cls+=' active';if(winner)cls+=' winner';if(done)cls+=' done';
  return '<div class="'+cls+'" '+(active?'aria-current="step"':'')+'><div class="mmr-live-card-head"><b>'+esc(x.id)+'</b>'+(winner?'<span>★ ĐƯỢC CHỌN</span>':active?'<span>ĐANG TÍNH</span>':'')+'</div><p>'+esc(x.text||'')+'</p>'+(opts.score!==undefined?'<div class="mmr-live-mini-score">MMRScore = '+num(opts.score,7)+'</div>':'')+(opts.detail||'')+'</div>';
}
function mmrLiveBoard(item){
  const pool=DATA.mmr.pool||[];
  if(!pool.length)return '<div class="mmr-live-empty">Centroid không trả về câu nào nên MMR không thể khởi tạo Q.</div>';
  const q=pool[0],rounds=DATA.mmr.rounds||[];
  const ri=Number.isInteger(item.roundIndex)?item.roundIndex:-1;
  const r=ri>=0?rounds[ri]:null;
  const candidates=r?.candidates||[];
  const winner=candidates.find(c=>c.winner)||candidates.find(c=>c.id===r?.winner_id)||null;
  const winnerStage=item.type==='mmr-winner'||(item.type==='mmr-loop-check'&&ri>=0);
  const activeIndex=Number.isInteger(item.candidateIndex)?item.candidateIndex:-1;
  const activeCandidate=activeIndex>=0?candidates[activeIndex]:(winnerStage?winner:null);
  const before=r?.selected_before||[q];
  const beforeU=r?.remaining_before||pool.slice(1);
  const after=winnerStage?(r?.selected_after||before):before;
  const afterU=winnerStage?(r?.remaining_after||beforeU):beforeU;
  const isNoRound=item.type==='mmr-loop-check'&&ri===-1;
  const phaseScore=item.type==='mmr-candidate-score'||item.type==='mmr-winner'||item.type==='mmr-loop-check';
  let currentStep='KHỞI TẠO R';
  if(item.type==='mmr-candidate-rel')currentStep='ĐANG TÍNH REL';
  else if(item.type==='mmr-candidate-red')currentStep='ĐANG TÍNH RED';
  else if(item.type==='mmr-candidate-score')currentStep='ĐANG TÍNH ĐIỂM MMR';
  else if(item.type==='mmr-winner')currentStep='CHỌN CÂU THẮNG';
  else if(item.type==='mmr-loop-check')currentStep='CẬP NHẬT R / KIỂM TRA DỪNG';
  const heading=ri>=0?'VÒNG '+(ri+1)+' · '+currentStep:currentStep;
  const chip=(x,opts={})=>mmrLiveCard(x,opts);
  const beforeHtml=before.map(x=>chip(x,{winner:winnerStage&&x.id===r?.winner_id})).join('')||'<p class="mmr-live-hint">R đang rỗng.</p>';
  const candidateHtml=beforeU.map(x=>{
    const ci=candidates.findIndex(c=>c.id===x.id);
    const c=ci>=0?candidates[ci]:null;
    const isActive=!!c&&ci===activeIndex;
    const isWinner=winnerStage&&x.id===r?.winner_id;
    const phaseForCard=isActive?item.type:null;
    const done=!!c&&(winnerStage||ci<activeIndex||(isActive&&phaseScore));
    const detail=isActive?mmrLiveCalc(item,c,r):(isWinner?mmrLiveCalc({type:'mmr-winner'},winner,r):'');
    return chip(x,{active:isActive,winner:isWinner,done:done&&!isWinner,score:done&&c?c.score:undefined,detail});
  }).join('')||'<p class="mmr-live-hint">Không còn ứng viên.</p>';
  const afterHtml=after.map(x=>chip(x,{winner:winnerStage&&x.id===r?.winner_id})).join('')||'<p class="mmr-live-hint">R đang rỗng.</p>';
  const remainingHtml=afterU.map(x=>{
    const c=candidates.find(z=>z.id===x.id);
    return chip(x,{score:winnerStage&&c?c.score:undefined,done:winnerStage&&!!c});
  }).join('')||'<p class="mmr-live-hint">Không còn ứng viên.</p>';
  const liveNote=isNoRound?'<div class="mmr-live-note warn"><b>Không có vòng MMR bổ sung.</b> Điều kiện dừng của chế độ hiện tại không cho phép lặp; giữ nguyên Q trong R. Hãy tăng số ứng viên qua Centroid để quan sát các lượt chọn.</div>':item.type==='mmr-loop-check'&&ri>=0?'<div class="mmr-live-note '+(ri<rounds.length-1?'good':'warn')+'"><b>'+(ri<rounds.length-1?'Tiếp tục vòng '+(ri+2):'Kết thúc MMR')+'.</b> R đã được cập nhật. Ở vòng tiếp theo, Rel so với Q vẫn được tính theo Q cố định; Red sẽ được tính lại theo tập R mới.</div>':'';
  return '<div class="mmr-live-shell"><div class="mmr-live-q"><div><b>Q · GIỮ NGUYÊN Q QUA MỌI VÒNG</b><small>'+esc(q.id)+' · '+esc(q.text)+'</small></div><span>'+esc(heading)+'</span></div><div class="mmr-live-grid"><section class="mmr-live-column before"><h3>TRƯỚC LƯỢT · R ĐÃ CHỌN</h3>'+beforeHtml+'<p class="mmr-live-column-note">Các câu trong R dùng để tính Red.</p></section><section class="mmr-live-column candidates"><h3>ỨNG VIÊN CHƯA CHỌN</h3>'+candidateHtml+'<p class="mmr-live-column-note">'+(activeCandidate?'Đang xét '+esc(activeCandidate.id)+'. Sau khi chấm hết ứng viên sẽ chọn MMRScore cao nhất.':'Chuyển từng bước để xem phép tính của từng câu.')+'</p></section><section class="mmr-live-column after"><h3>SAU LƯỢT · R MỚI</h3>'+afterHtml+'<p class="mmr-live-column-note">'+(winnerStage&&winner?'Đã thêm '+esc(winner.id)+' vào R.':'Chưa chuyển câu thắng cho đến Bước 4.')+'</p></section><section class="mmr-live-column remaining"><h3>ỨNG VIÊN CÒN LẠI</h3>'+remainingHtml+'<p class="mmr-live-column-note">'+(winnerStage?'Câu thắng đã bị loại khỏi U.':'Sau khi chọn câu thắng, câu đó sẽ bị bỏ khỏi danh sách này.')+'</p></section></div>'+liveNote+'</div>';
}

function flowRenderContent(item){if(item.group===4)return mmrLiveBoard(item);switch(item.type){case'input':return flowInput();case'preprocess':return flowPreprocess();case'wordvec':return flowWordVec();case'km-points':return kmPointsStep();case'km-init':return kmInitStep();case'km-frame':return kmTeachingFrameStep(item);case'km-move':return kmMovementStep(item);case'km-final':return kmeansFinalConnections();case'km-animation':return kmAnimationStep();case'representative-summary':return representativeSummaryStep();case'representative':return representativeStep(item);case'centroid-overview':return centroidOverviewStep();case'centroid-vector':return centroidVectorStep();case'centroid-candidate':return centroidCandidateScore(item);case'centroid-ranking':return centroidRankingStep();case'centroid-decision':return centroidDecisionStep(item);case'mmr-empty':return '<div class="note warn">Không có ứng viên sau CentroidBow nên không thể khởi tạo Q cho MMR. Hãy kiểm tra ngưỡng hoặc văn bản đầu vào.</div>';case'mmr-query':return mmrQueryStep();case'mmr-rel-overview':return mmrRelOverviewStep(item);case'mmr-red-overview':return mmrRedOverviewStep(item);case'mmr-score-overview':return mmrScoreOverviewStep(item);case'mmr-candidate-evaluate':return mmrScoreStep(item);case'mmr-winner':return mmrWinnerStep(item);case'mmr-loop-check':return mmrLoopCheckStep(item);case'mmr-order':return '<div class="flow-focus"><b>Thứ tự theo điểm MMR</b>MMR thêm từng câu theo thứ tự thắng của các vòng.</div><ol class="final-list">'+DATA.mmr.selected.map((x,i)=>'<li><small>'+String(i+1)+'. '+esc(x.id)+'</small>'+esc(x.text)+'</li>').join('')+'</ol>';case'final':return finalStage();default:return '<div class="note">Không có nội dung cho bước này.</div>';}}
function renderFlowStrip(item){
  const best=(DATA.kmeans.restarts||[]).find(r=>r.best), frame=item.runIndex!==undefined?runFor(item).history?.[item.frameIndex]:null;
  const repTexts=new Set(),centTexts=new Set(),mmrTexts=new Set((DATA.mmr.selected||[]).map(x=>x.text)),finalTexts=new Set((DATA.final||[]).map(x=>x.text));
  // Reveal accepted representatives incrementally instead of showing future selections up front.
  const repSteps=DATA.representative_steps||[];
  if(item.group===2){
    if(item.type==='representative-summary'){
      repSteps.forEach(x=>{if(x.selected_text)repTexts.add(x.selected_text);});
    }else{
      const through=item.representativeIndex===undefined?-1:item.representativeIndex;
      repSteps.slice(0,through).forEach(x=>{if(x.selected_text)repTexts.add(x.selected_text);});
      const active=repSteps[through];if(active?.selected_text)repTexts.add(active.selected_text);
    }
  }
  // Centroid decisions are revealed in descending cosine order, only through the active candidate.
  const centroidItems=DATA.centroid.process||[];
  const centroidRanked=(DATA.centroid.ranking||[]).slice().sort((a,b)=>(Number(b.cosine_score)-Number(a.cosine_score))||(Number(a.index)-Number(b.index)));
  if(item.group>3) (DATA.centroid.selected||[]).forEach(x=>centTexts.add(x.text));
  else if(item.group===3 && item.type==='centroid-decision'){
    const active=centroidItems[item.centroidIndex];
    const activeRank=active?centroidRanked.findIndex(r=>Number(r.index)===Number(active.index)):-1;
    centroidRanked.slice(0,activeRank+1).forEach(r=>{
      const candidate=centroidItems.find(x=>Number(x.index)===Number(r.index));
      if(candidate?.selected)centTexts.add(candidate.text);
    });
  }
  // MMR highlights the live selected set R: before transfer use selected_before; after it use selected_after.
  const mmrPickedTexts=new Set();
  if(item.group>4) mmrTexts.forEach(t=>mmrPickedTexts.add(t));
  else if(item.group===4 && (item.type==='mmr-query'||(item.type==='mmr-loop-check'&&item.roundIndex===-1))){
    if(DATA.mmr.pool?.[0]?.text)mmrPickedTexts.add(DATA.mmr.pool[0].text);
  }else if(item.group===4 && item.roundIndex!==undefined){
    const round=DATA.mmr.rounds?.[item.roundIndex];
    const liveR=(item.type==='mmr-winner'||item.type==='mmr-loop-check')?(round?.selected_after||[]):(round?.selected_before||[]);
    liveR.forEach(x=>mmrPickedTexts.add(x.text));
  }
  const finalPicked=item.group>=5?finalTexts:new Set();
  const focusedText= item.sentenceIndex!==undefined?DATA.sentences[item.sentenceIndex]?.text:item.seedIndex!==undefined?DATA.sentences[item.seedIndex]?.text:item.representativeIndex!==undefined?DATA.representative_steps?.[item.representativeIndex]?.selected_text:item.centroidIndex!==undefined?DATA.centroid.process?.[item.centroidIndex]?.text:item.candidateIndex!==undefined?DATA.mmr.rounds?.[item.roundIndex]?.candidates?.[item.candidateIndex]?.text:null;
  $('flow-strip').innerHTML=DATA.sentences.map((s,i)=>{
    let cluster=null;const kLive=item.type==='km-frame'&&kmLessonState&&kmLessonState.item===item;
    if(item.group>=1&&item.type!=='km-points'&&item.type!=='km-init'){
      if(kLive){const st=kmLessonState,idx=Number(s.index??i);if(st.phase!=='sweep'||i<st.cursor||(i===st.cursor&&st.pointPhase==='assigned'))cluster=(st.phase==='complete'?st.frame.post_assignments?.[idx]:null)??st.frame.assignments?.[idx];}
      else if(frame&&frame.assignments)cluster=frame.assignments[i];else if(best?.history?.length)cluster=best.history[best.history.length-1].assignments[i];
    }
    let bg=cluster===null?'#aab3c1':COLORS[cluster%COLORS.length];
    let picked=false;
    if(item.group===2)picked=repTexts.has(s.text);
    else if(item.group===3)picked=centTexts.has(s.text);
    else if(item.group===4)picked=mmrPickedTexts.has(s.text);
    else if(item.group===5&&item.type==='mmr-order')picked=mmrTexts.has(s.text);
    else if(item.group>=5)picked=finalTexts.has(s.text);
    if(picked)bg='#866aae';
    const active=(kLive&&kmLessonState.phase==='sweep'&&i===kmLessonState.cursor)||focusedText===s.text;
    return '<div class="flow-tile '+(active?'active ':'')+(picked?'selected':'')+'" style="background:'+bg+'" title="'+esc(s.text)+'">'+esc(s.id)+(picked?' ★':'')+'<small>'+esc(s.text)+'</small></div>';
  }).join('');
}

function lessonBarRows(items,key,label,format='score'){
  const vals=items.map(x=>Number(x[key])).filter(Number.isFinite),lo=vals.length?Math.min(...vals):0,hi=vals.length?Math.max(...vals):1,range=(hi-lo)||1;
  return '<div class="lecture-bars">'+items.map((x,i)=>{const v=Number(x[key]),valid=Number.isFinite(v),width=valid?Math.max(2,100*(v-lo)/range):2;return '<div class="lecture-bar-row"><div class="lecture-bar-label" title="'+esc(x.id||x.term||'')+'">'+esc(x.id||x.term||('Mục '+(i+1)))+'</div><div class="lecture-track"><div class="lecture-fill '+(x.winner?'winner':(key==='red'?'penalty':''))+'" style="width:'+Math.max(2,Math.min(100,width))+'%"></div></div><div class="lecture-bar-value">'+num(v,4)+'</div></div>';}).join('')+'</div>';
}
function lessonBoard(item){
  // The step title and main visualization already appear in the stage card above.
  // Keep this compatibility hook empty to avoid duplicate previews and narration.
  return '';
}
let flowIsPlaying=false;
function flowSchedulePlay(){
  if(!flowIsPlaying)return;
  const item=flowSteps[flowIndex],delay=Number($('flow-speed')?.value||3000);
  const advance=()=>{
    if(!flowIsPlaying)return;
    if(flowIndex>=flowSteps.length-1){flowIsPlaying=false;flowTimer=null;$('flow-play').textContent='↻ Chạy lại';$('flow-live').textContent='HOÀN TẤT';return;}
    flowGo(flowIndex+1);$('flow-live').textContent='ĐANG CHẠY';flowSchedulePlay();
  };
  const waitUntilSceneDone=()=>{
    if(!flowIsPlaying)return;
    if(item?.type==='km-frame'&&kmLessonState&&!['complete'].includes(kmLessonState.phase)){flowTimer=setTimeout(waitUntilSceneDone,160);return;}
    if(item?.type==='km-init'&&kmInitState&&(kmInitState.running||kmInitState.cursor<(kmInitState.frame.centers||[]).length-1)){flowTimer=setTimeout(waitUntilSceneDone,160);return;}
    if(item?.type==='km-move'&&kmMoveState){
      const k=(kmMoveState.frame.centers_after||kmMoveState.frame.centers||[]).length;
      if(!kmMoveState.running&&kmMoveState.clusterIndex<k-1)kmMovePlay();
      if(kmMoveState.running||kmMoveState.clusterIndex<k-1){flowTimer=setTimeout(waitUntilSceneDone,160);return;}
    }
    flowTimer=setTimeout(advance,450);
  };
  const kmMinimum=item?.type==='km-frame'?Math.min(45000,Math.max(delay,(DATA.sentences.length*650)+1900)):(item?.type==='km-move'?Math.max(delay,((DATA.kmeans.k||1)*1050)+700):delay);
  flowTimer=setTimeout(waitUntilSceneDone,kmMinimum);
}
function flowTogglePlay(){
  if(flowIsPlaying){
    flowIsPlaying=false;if(flowTimer){clearTimeout(flowTimer);flowTimer=null;}
    if(kmLessonState?.running){kmLessonState.running=false;kmLessonStopTimer();kmLessonRefresh();}
    if(kmInitState?.running){kmInitState.running=false;kmInitStop();kmInitRefresh();}
    if(kmMoveState?.running){kmMoveState.running=false;kmMoveStopTimer();kmMoveRefresh();}
    $('flow-play').textContent='▶ Tự chạy';$('flow-live').textContent='ĐÃ TẠM DỪNG';return;
  }
  if(flowIndex>=flowSteps.length-1)flowGo(0);
  if(kmLessonState&&!kmLessonState.running&&kmLessonState.phase!=='complete')kmLessonPlay();
  flowIsPlaying=true;$('flow-play').textContent='⏸ Tạm dừng';$('flow-live').textContent='ĐANG CHẠY';flowSchedulePlay();
}

function flowRender(){if(!DATA||!flowSteps.length)return;stopKmGraph();kmLessonStopTimer();kmInitStop();kmMoveStopTimer();kmInitState=null;kmLessonState=null;kmMoveState=null;const item=flowSteps[flowIndex];const mmrOnlyView=item.group===4;const teachingHeading=document.querySelector('.flow-step-heading');const sentenceStrip=document.querySelector('.flow-strip');if(teachingHeading)teachingHeading.style.display=mmrOnlyView?'none':'';if(sentenceStrip)sentenceStrip.style.display=mmrOnlyView?'none':'';$('flow-groups').innerHTML=FLOW_GROUPS.map((label,i)=>'<button type="button" data-flow-group="'+i+'" class="'+(i===item.group?'on':'')+'"><span class="flow-group-number">'+String(i+1).padStart(2,'0')+'</span><span>'+label+'</span></button>').join('');$('flow-groups').querySelectorAll('[data-flow-group]').forEach(b=>b.onclick=()=>{flowStop();const g=Number(b.dataset.flowGroup),first=flowSteps.findIndex(x=>x.group===g);if(first>=0){flowGo(first);}});$('flow-badge').textContent=String(item.group+1).padStart(2,'0');$('flow-kicker').textContent='GIAI ĐOẠN '+String(item.group+1).padStart(2,'0')+' / '+FLOW_GROUPS.length+' · BƯỚC '+String(flowIndex+1).padStart(2,'0')+' / '+String(flowSteps.length).padStart(2,'0');$('flow-title').textContent=item.title;$('flow-description').textContent=item.description;$('flow-content').innerHTML=lessonBoard(item)+flowRenderContent(item);$('flow-content').className='flow-content-enter type-'+item.type;void $('flow-content').offsetWidth;$('flow-count').textContent='Bước '+(flowIndex+1)+' / '+flowSteps.length;$('flow-progress').max=Math.max(1,flowSteps.length-1);$('flow-progress').value=flowIndex;$('flow-prev').disabled=flowIndex===0;$('flow-next').disabled=flowIndex===flowSteps.length-1;$('flow-live').textContent='ĐANG XEM';renderFlowStrip(item);if(item.type==='km-animation')bindKmGraphControls();if(item.type==='km-init')bindKmInitControls();if(item.type==='km-frame')bindKmLessonControls(item);if(item.type==='km-move')bindKmMoveControls(item);}
function flowGo(n){flowIndex=Math.max(0,Math.min(n,flowSteps.length-1));flowRender();}
function flowStop(){flowIsPlaying=false;if(flowTimer){clearTimeout(flowTimer);flowTimer=null;}stopKmGraph();kmInitStop();kmMoveStopTimer();if(kmMoveState)kmMoveState.running=false;if(kmInitState)kmInitState.running=false;$('flow-play').textContent='▶ Tự chạy';$('flow-live').textContent='ĐÃ TẠM DỪNG';}
$('flow-prev').onclick=()=>{flowStop();flowGo(flowIndex-1);};$('flow-next').onclick=()=>{flowStop();flowGo(flowIndex+1);};$('flow-speed').onchange=()=>{if(flowIsPlaying){if(flowTimer){clearTimeout(flowTimer);flowTimer=null;}flowSchedulePlay();}};$('flow-play').onclick=flowTogglePlay;
document.addEventListener('keydown',e=>{if(!$('results')||$('results').style.display==='none')return;if(e.key==='ArrowRight'){flowStop();flowGo(flowIndex+1)}if(e.key==='ArrowLeft'){flowStop();flowGo(flowIndex-1)}});
function render(){if(!DATA)return;flowBuild();flowIndex=0;flowRender();}

$('run').onclick=async()=>{const button=$('run');button.disabled=true;button.textContent='Đang tính…';$('status').className='status';try{const body={documents:docs(),clusters:$('clusters').value==='auto'?null:Number($('custom-k').value),mode:$('mode').value,lambda:Number($('lambda').value),centroid_mode:'cosine',cosine_threshold:cosineInputThreshold()};RUN_DOCS=body.documents.slice();const res=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const payload=await res.json();if(!res.ok)throw new Error(payload.error||'Không thể chạy pipeline.');DATA=payload;stage=0;kmRestart=Math.max(0,Number(DATA.kmeans.best_run||1)-1);kmStep=0;centroidStep=0;mmrStep=0;kmAnimIndex=0;flowStop();$('input-details').open=false;$('empty').style.display='none';$('results').style.display='block';render();status('Đã tính xong.','info');}catch(e){status(e.message,'error');}finally{button.disabled=false;button.textContent='Chạy pipeline';}};

</script></body></html>'''





def _json_number(value: Any) -> float | str | None:

    """Convert numpy/Python numbers to values accepted by strict JSON."""

    try:

        number = float(value)

    except (TypeError, ValueError):

        return None

    if math.isnan(number):

        return "NaN"

    if math.isinf(number):

        return "∞" if number > 0 else "−∞"

    return number





def _same_number(a: Any, b: Any) -> bool:
    """Compare finite values and matching infinities/NaNs for the MMR audit."""
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if math.isnan(x) and math.isnan(y):
        return True
    if x == y:  # includes equal positive/negative infinities
        return True
    return math.isfinite(x) and math.isfinite(y) and math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-10)


def _distances(X, centers):

    import numpy as np



    delta = X[:, None, :] - centers[None, :, :]

    return np.einsum("nkd,nkd->nk", delta, delta)





def _lloyd_trace_portable(X, initial_centers, max_iter, tolerance):
    """Portable Lloyd trace fallback for sklearn versions without its Cython helper."""
    import numpy as np

    centers = np.asarray(initial_centers).copy()
    n_samples = len(X)
    n_clusters = len(centers)
    history = []
    previous_labels = None

    def members_for(labels):
        return [np.flatnonzero(labels == j).astype(int).tolist() for j in range(n_clusters)]

    initial_matrix = _distances(X, centers)
    initial_labels = np.argmin(initial_matrix, axis=1)
    initial_chosen = initial_matrix[np.arange(n_samples), initial_labels]
    history.append({
        "iteration": 0, "centers": centers.copy(), "centers_after": centers.copy(),
        "assignments": initial_labels.copy(), "distance_matrix": initial_matrix.copy(),
        "distances": initial_chosen.copy(), "post_assignments": initial_labels.copy(),
        "post_distances": initial_chosen.copy(), "inertia": float(initial_chosen.sum()),
        "label": "Khởi tạo K-means++",
        "note": "Các tâm ban đầu được chọn từ vector Word2Vec; chưa cập nhật tâm.",
        "shift": None, "cluster_updates": [], "converged": False,
        "convergence_reason": "Chưa lặp Lloyd",
    })

    for iteration in range(max_iter):
        centers_before = centers.copy()
        distance_matrix = _distances(X, centers_before)
        labels = np.argmin(distance_matrix, axis=1)
        counts = np.bincount(labels, minlength=n_clusters).astype(float)
        sums = np.zeros_like(centers_before)
        np.add.at(sums, labels, X)
        update_members = members_for(labels)
        relocated = []

        empty = np.flatnonzero(counts == 0)
        if len(empty):
            assigned_distances = ((X - centers_before[labels]) ** 2).sum(axis=1)
            if float(np.max(assigned_distances)) > 0.0:
                farthest = np.argpartition(assigned_distances, -len(empty))[:-len(empty)-1:-1]
            else:
                farthest = np.asarray([], dtype=int)
            for empty_cluster, candidate_index in zip(empty, farthest):
                sample_index = int(candidate_index)
                old_cluster = int(labels[sample_index])
                sums[old_cluster] -= X[sample_index]
                sums[empty_cluster] += X[sample_index]
                counts[old_cluster] -= 1
                counts[empty_cluster] += 1
                if sample_index in update_members[old_cluster]:
                    update_members[old_cluster].remove(sample_index)
                update_members[int(empty_cluster)].append(sample_index)
                relocated.append({"sample_index": sample_index, "from_cluster": old_cluster, "to_cluster": int(empty_cluster)})

        centers_after = centers_before.copy()
        nonempty = counts > 0
        centers_after[nonempty] = sums[nonempty] / counts[nonempty, None]
        if np.any(~nonempty) and np.any(nonempty):
            centers_after[~nonempty] = centers_after[int(np.argmax(counts))]

        shift_by_cluster = np.sum((centers_after - centers_before) ** 2, axis=1)
        shift = float(shift_by_cluster.sum())
        post_matrix = _distances(X, centers_after)
        post_labels = np.argmin(post_matrix, axis=1)
        post_distances = post_matrix[np.arange(n_samples), post_labels]
        strict = previous_labels is not None and np.array_equal(labels, previous_labels)
        tolerance_reached = shift <= tolerance
        converged = bool(strict or tolerance_reached)
        reason = ("Nhãn cụm không đổi so với vòng trước." if strict else
                  f"Tổng bình phương độ dịch tâm {shift:.6g} ≤ tolerance {tolerance:.6g}." if tolerance_reached else
                  f"Chưa hội tụ: tổng bình phương độ dịch tâm {shift:.6g} > tolerance {tolerance:.6g}.")
        cluster_updates = []
        for j in range(n_clusters):
            cluster_updates.append({
                "cluster": j, "members": update_members[j].copy(), "count": int(round(counts[j])),
                "old_preview": centers_before[j][:8].copy(), "new_preview": centers_after[j][:8].copy(),
                "old_vector": centers_before[j].copy(), "new_vector": centers_after[j].copy(),
                "shift_sq": float(shift_by_cluster[j]),
                "relocated": [x for x in relocated if x["to_cluster"] == j or x["from_cluster"] == j],
            })
        note = "Gán câu vào tâm gần nhất, sau đó cập nhật tâm bằng trung bình vector của các câu trong cụm. " + reason
        if relocated:
            note += " Có mẫu được chuyển để xử lý cụm rỗng."
        elif len(empty):
            note += " Khoảng cách đều bằng 0; tâm rỗng được đặt trùng tâm cụm lớn nhất."
        history.append({
            "iteration": iteration + 1, "centers": centers_before, "centers_after": centers_after.copy(),
            "assignments": labels.copy(), "distance_matrix": distance_matrix.copy(),
            "distances": distance_matrix[np.arange(n_samples), labels].copy(),
            "post_assignments": post_labels.copy(), "post_distances": post_distances.copy(),
            "inertia": float(post_distances.sum()), "label": "Vòng Lloyd " + str(iteration + 1),
            "note": note, "shift": shift, "cluster_updates": cluster_updates,
            "converged": converged, "convergence_reason": reason,
        })
        centers = centers_after
        if converged:
            break
        previous_labels = labels.copy()

    final_distances = _distances(X, centers)
    final_labels = np.argmin(final_distances, axis=1)
    return history, centers, final_labels, float(final_distances[np.arange(n_samples), final_labels].sum())


def _lloyd_trace(X, initial_centers, max_iter, tolerance):
    """Trace the real sklearn dense-Lloyd iteration when its helper is available.

    Calling the same low-level Lloyd iteration function used by KMeans.fit avoids
    differences in empty-cluster relocation and floating-point assignment ties.
    A portable, transparent implementation remains as a compatibility fallback.
    """
    import numpy as np

    try:
        from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_dense
        from sklearn.cluster._k_means_common import _inertia_dense
        from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
    except ImportError:
        return _lloyd_trace_portable(X, initial_centers, max_iter, tolerance)

    X = np.ascontiguousarray(X)
    centers = np.ascontiguousarray(initial_centers, dtype=X.dtype).copy()
    n_samples, n_features = X.shape
    n_clusters = centers.shape[0]
    sample_weight = np.ones(n_samples, dtype=X.dtype)
    centers_new = np.zeros_like(centers)
    labels = np.full(n_samples, -1, dtype=np.int32)
    labels_old = labels.copy()
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)
    n_threads = _openmp_effective_n_threads()
    history = []

    def make_members(current_labels):
        return [np.flatnonzero(current_labels == j).astype(int).tolist() for j in range(n_clusters)]

    def assign_only(current_centers):
        current_labels = np.full(n_samples, -1, dtype=np.int32)
        current_weights = np.zeros(n_clusters, dtype=X.dtype)
        current_shift = np.zeros(n_clusters, dtype=X.dtype)
        lloyd_iter_chunked_dense(
            X, sample_weight, current_centers, current_centers,
            current_weights, current_labels, current_shift, n_threads,
            update_centers=False,
        )
        return current_labels

    initial_matrix = _distances(X, centers)
    initial_labels = assign_only(centers)
    initial_chosen = initial_matrix[np.arange(n_samples), initial_labels]
    history.append({
        "iteration": 0,
        "centers": centers.copy(),
        "centers_after": centers.copy(),
        "assignments": initial_labels.copy(),
        "distance_matrix": initial_matrix.copy(),
        "distances": initial_chosen.copy(),
        "post_assignments": initial_labels.copy(),
        "post_distances": initial_chosen.copy(),
        "inertia": float(_inertia_dense(X, sample_weight, centers, initial_labels, n_threads)),
        "label": "Khởi tạo K-means++",
        "note": "Đã chọn K tâm ban đầu bằng K-means++. Các phép gán/đổi tâm ở những vòng dưới sử dụng cùng hàm Lloyd nội bộ của scikit-learn.",
        "shift": None,
        "cluster_updates": [],
        "converged": False,
        "convergence_reason": "Chưa lặp Lloyd",
    })

    converged = False
    strict_convergence = False
    for iteration in range(max_iter):
        centers_before = centers.copy()
        distance_matrix = _distances(X, centers_before)

        # This is exactly the assignment + sum/count + empty-cluster relocation
        # + averaging step used by sklearn's dense Lloyd implementation.
        lloyd_iter_chunked_dense(
            X, sample_weight, centers, centers_new,
            weight_in_clusters, labels, center_shift, n_threads,
        )
        assignments = labels.copy()
        centers_after = centers_new.copy()
        counts = weight_in_clusters.copy()

        # Reconstruct which examples sklearn relocates only for explaining the
        # centroid arithmetic. The centers themselves come from sklearn's helper.
        update_members = make_members(assignments)
        relocated = []
        empty_clusters = np.flatnonzero(np.bincount(assignments, minlength=n_clusters) == 0)
        if len(empty_clusters):
            assigned_distances = ((X - centers_before[assignments]) ** 2).sum(axis=1)
            if float(np.max(assigned_distances)) > 0.0:
                farthest = np.argpartition(assigned_distances, -len(empty_clusters))[:-len(empty_clusters)-1:-1]
            else:
                farthest = np.asarray([], dtype=int)
            used = set()
            for empty_cluster, candidate_index in zip(empty_clusters, farthest):
                if int(candidate_index) in used:
                    candidate_index = next(index for index in farthest if int(index) not in used)
                sample_index = int(candidate_index)
                used.add(sample_index)
                old_cluster = int(assignments[sample_index])
                if sample_index in update_members[old_cluster]:
                    update_members[old_cluster].remove(sample_index)
                update_members[int(empty_cluster)].append(sample_index)
                relocated.append({
                    "sample_index": sample_index,
                    "from_cluster": old_cluster,
                    "to_cluster": int(empty_cluster),
                })

        shift_by_cluster = np.sum((centers_after - centers_before) ** 2, axis=1)
        # sklearn checks the squares of the Cython-computed per-center shifts.
        shift = float(np.sum(center_shift ** 2))
        post_labels = assign_only(centers_after)
        post_matrix = _distances(X, centers_after)
        post_distances = post_matrix[np.arange(n_samples), post_labels]
        strict_convergence = bool(np.array_equal(assignments, labels_old))
        tolerance_convergence = shift <= tolerance
        converged = bool(strict_convergence or tolerance_convergence)
        if strict_convergence:
            reason = "Nhãn gán cụm không đổi so với vòng Lloyd trước (hội tụ nghiêm ngặt)."
        elif tolerance_convergence:
            reason = f"Tổng bình phương độ dịch tâm {shift:.6g} ≤ tolerance {tolerance:.6g}."
        else:
            reason = f"Chưa hội tụ: tổng bình phương độ dịch tâm {shift:.6g} > tolerance {tolerance:.6g}."

        cluster_updates = []
        for cluster_index in range(n_clusters):
            cluster_updates.append({
                "cluster": cluster_index,
                "members": update_members[cluster_index].copy(),
                "count": int(round(float(counts[cluster_index]))),
                "old_preview": centers_before[cluster_index][:8].copy(),
                "new_preview": centers_after[cluster_index][:8].copy(),
                "old_vector": centers_before[cluster_index].copy(),
                "new_vector": centers_after[cluster_index].copy(),
                "shift_sq": float(shift_by_cluster[cluster_index]),
                "relocated": [x for x in relocated if x["to_cluster"] == cluster_index or x["from_cluster"] == cluster_index],
            })

        note = "Bước 1: gán từng câu vào tâm gần nhất. Bước 2: tính trung bình vector để cập nhật tâm; khi gặp cụm rỗng, dùng quy tắc xử lý cụm rỗng của scikit-learn. " + reason
        if relocated:
            note += " Có mẫu được chuyển tạm để cập nhật cụm rỗng; danh sách chuyển được ghi trong bảng cập nhật tâm."
        elif len(empty_clusters):
            note += " Có cụm rỗng và tất cả khoảng cách gán đều bằng 0; scikit-learn đặt tâm rỗng trùng tâm cụm lớn nhất."
        history.append({
            "iteration": iteration + 1,
            "centers": centers_before,
            "centers_after": centers_after,
            "assignments": assignments,
            "distance_matrix": distance_matrix,
            "distances": distance_matrix[np.arange(n_samples), assignments],
            "post_assignments": post_labels.copy(),
            "post_distances": post_distances.copy(),
            "inertia": float(_inertia_dense(X, sample_weight, centers_after, post_labels, n_threads)),
            "label": "Vòng Lloyd " + str(iteration + 1),
            "note": note,
            "shift": shift,
            "cluster_updates": cluster_updates,
            "converged": converged,
            "convergence_reason": reason,
        })

        centers, centers_new = centers_new, centers
        if converged:
            break
        labels_old[:] = assignments

    # KMeans.fit performs one final assignment pass if it stopped due to
    # tolerance rather than unchanged labels; mirror that for the returned state.
    if not strict_convergence:
        lloyd_iter_chunked_dense(
            X, sample_weight, centers, centers_new,
            weight_in_clusters, labels, center_shift, n_threads,
            update_centers=False,
        )
    final_labels = labels.copy()
    final_inertia = float(_inertia_dense(X, sample_weight, centers, final_labels, n_threads))
    return history, centers, final_labels, final_inertia


def _kmeans_plusplus_trace(X, n_clusters, random_state):
    """Replay sklearn K-means++ and record the actual D² sampling trials.

    The calculation mirrors sklearn's private initializer for dense, equally
    weighted sentence vectors. The caller compares the selected seeds with the
    public sklearn function before using this trace in the UI.
    """
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.utils.extmath import row_norms

    n_samples, n_features = X.shape
    sample_weight = np.ones(n_samples, dtype=X.dtype)
    x_squared_norms = row_norms(X, squared=True)
    n_local_trials = 2 + int(np.log(n_clusters)) if n_clusters > 1 else 1
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    indices = np.full(n_clusters, -1, dtype=int)
    trace = []

    first_index = int(random_state.choice(n_samples, p=sample_weight / sample_weight.sum()))
    centers[0] = X[first_index]
    indices[0] = first_index
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
    ).ravel()
    current_potential = float(closest_dist_sq @ sample_weight)
    trace.append({
        "center_index": 0,
        "selected_index": first_index,
        "first": True,
        "current_potential": current_potential,
        "n_local_trials": 1,
        "probabilities": [{
            "index": i,
            "distance_sq": None,
            "distance": None,
            "probability": float(sample_weight[i] / sample_weight.sum()),
        } for i in range(n_samples)],
        "trials": [{
            "candidate_index": first_index,
            "target": None,
            "sample_probability": float(sample_weight[first_index] / sample_weight.sum()),
            "potential": current_potential,
            "chosen": True,
        }],
    })

    for center_index in range(1, n_clusters):
        weights = sample_weight * closest_dist_sq
        total_weight = float(weights.sum())
        probabilities = weights / total_weight if total_weight > 0 else np.zeros_like(weights)
        potential_before = current_potential
        random_targets = random_state.uniform(size=n_local_trials) * current_potential
        candidate_indices = np.searchsorted(np.cumsum(weights), random_targets)
        np.clip(candidate_indices, None, closest_dist_sq.size - 1, out=candidate_indices)

        distance_to_candidates = euclidean_distances(
            X[candidate_indices], X, Y_norm_squared=x_squared_norms, squared=True
        )
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidate_potentials = (distance_to_candidates @ sample_weight.reshape(-1, 1)).ravel()
        best_trial = int(np.argmin(candidate_potentials))
        selected_index = int(candidate_indices[best_trial])
        current_potential = float(candidate_potentials[best_trial])
        closest_dist_sq = distance_to_candidates[best_trial].copy()

        trace.append({
            "center_index": int(center_index),
            "selected_index": selected_index,
            "first": False,
            "current_potential": potential_before,
            "remaining_potential": current_potential,
            "n_local_trials": int(n_local_trials),
            "probabilities": [{
                "index": i,
                "distance_sq": _json_number(float(weights[i])),
                "distance": _json_number(float(np.sqrt(max(float(weights[i]), 0.0)))),
                "probability": _json_number(float(probabilities[i])),
            } for i in range(n_samples)],
            "trials": [{
                "candidate_index": int(candidate_indices[j]),
                "target": _json_number(float(random_targets[j])),
                "sample_probability": _json_number(float(probabilities[int(candidate_indices[j])])),
                "potential": _json_number(float(candidate_potentials[j])),
                "chosen": bool(j == best_trial),
            } for j in range(len(candidate_indices))],
        })
        centers[center_index] = X[selected_index]
        indices[center_index] = selected_index

    return centers, indices, trace


def _effective_n_init(estimator):

    value = estimator.n_init

    if value == "auto":

        return 1 if estimator.init == "k-means++" or hasattr(estimator.init, "shape") else 10

    return int(value)





def _sentence_similarity_detail(sentence1, sentence2, idf):

    """Explain the exact asymmetric MMR.sentenceSim implementation."""

    import math as _math



    freq1 = sentence1.getWordFreqs()

    freq2 = sentence2.getWordFreqs()

    numerator = 0.0

    contributions = {}

    for word in sentence2.getProcessedWords():

        contribution = freq1.get(word, 0) * freq2.get(word, 0) * idf.get(word, 0) ** 2

        numerator += contribution

        if freq1.get(word, 0):

            contributions[word] = contributions.get(word, 0.0) + contribution

    denominator_sq = 0.0

    for word in sentence1.getProcessedWords():

        denominator_sq += (freq1.get(word, 0) * idf.get(word, 0)) ** 2

    denominator = _math.sqrt(denominator_sq)

    try:

        score = numerator / denominator

    except ZeroDivisionError:

        score = float("-inf")

    terms = [{

        "word": word,

        "tf1": freq1.get(word, 0),

        "tf2": freq2.get(word, 0),

        "idf": _json_number(idf.get(word, 0)),

        "contribution": _json_number(value),

    } for word, value in contributions.items()]

    denominator_terms = [{

        "word": word,

        "tf1": freq1.get(word, 0),

        "idf": _json_number(idf.get(word, 0)),

        "contribution": _json_number(freq1.get(word, 0) * (freq1.get(word, 0) * idf.get(word, 0)) ** 2),

    } for word in freq1]

    return {

        "numerator": _json_number(numerator),

        "denominator": _json_number(denominator),

        "score": _json_number(score),

        "terms": terms,

        "denominator_terms": denominator_terms,

    }





def run_pipeline(payload):

    import random
    import numpy as np

    # Cố định seed mỗi lần gọi API để cùng đầu vào cho kết quả tái lập được.
    # Word2Vec vẫn cần seed riêng và workers=1 để tránh bất định do đa luồng.
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    from gensim.models import Word2Vec

    from sklearn.cluster import KMeans, kmeans_plusplus

    from sklearn.decomposition import PCA

    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    from sklearn.metrics import pairwise_distances_argmin_min

    from underthesea import sent_tokenize, word_tokenize



    from methods.centroid.centroid_summarizer import CentroidBow

    from methods.main_method.Kmeans_CentroidBase_MMR_SentencePosition import Summarizer, mmr_target_count

    from methods.maximal_marginal_relevance.mmr_summarizer import MMR

    from utils.sentence import Sentence



    documents = payload.get("documents") or []

    if not isinstance(documents, list) or len(documents) > 30:

        raise ValueError("Nhập tối đa 30 tài liệu.")

    if any(not isinstance(x, str) for x in documents):

        raise ValueError("Nội dung mỗi tài liệu phải là văn bản.")

    if sum(len(x) for x in documents) > 100000:

        raise ValueError("Tổng nội dung tối đa 100.000 ký tự.")

    mode = payload.get("mode", "train")

    if mode not in ("train", "test"):

        raise ValueError("Chế độ MMR không hợp lệ.")

    lambta = float(payload.get("lambda", 0.6))

    if not 0 <= lambta <= 1:

        raise ValueError("λ phải nằm trong khoảng 0 đến 1.")

    centroid_mode = payload.get("centroid_mode", "cosine")
    if centroid_mode not in ("cosine", "repository"):
        raise ValueError("Chế độ Centroid không hợp lệ.")
    cosine_threshold = float(payload.get("cosine_threshold", 0.35))
    if not 0 <= cosine_threshold <= 1:
        raise ValueError("Ngưỡng cosine phải nằm trong khoảng 0 đến 1.")



    sentences = []

    boundaries = []

    source_doc = []
    source_positions = []

    dropped_count = 0

    punctuation = [".", ",", ";", ":", "!", "?", "(", ")", "`", "_", "--", "``", "''", "'s"]

    for document_index, text in enumerate(documents):

        boundaries.append(len(sentences))

        for input_position, original in enumerate(sent_tokenize(text.strip())):

            original = original.strip()

            if not original:

                continue

            tokenized = word_tokenize(original.lower(), format="text")

            processed = tokenized.split() if isinstance(tokenized, str) else list(tokenized)

            processed = [word for word in processed if word not in punctuation and "&" not in word]

            if len(processed) <= 4 or not processed:

                dropped_count += 1

                continue

            sentences.append(Sentence("doc_" + str(document_index + 1), processed, original))

            source_doc.append(document_index)
            source_positions.append(input_position)



    if len(sentences) < 5:

        raise ValueError("Cần ít nhất 5 câu hợp lệ sau tiền xử lý (mỗi câu phải có hơn 4 token). Hiện có " + str(len(sentences)) + ".")

    if len(sentences) > 300:

        raise ValueError("Bản demo giới hạn 300 câu để biểu đồ và phép tính K-means chạy gọn.")



    text_sents = [sentence.getProcessedWords() for sentence in sentences]

    org_sents = [sentence.getOGwords() for sentence in sentences]

    sentence_count = len(sentences)

    automatic_k = 3 if sentence_count < 20 else 5 if sentence_count < 50 else 8 if sentence_count < 100 else 10

    requested_k = payload.get("clusters")

    if requested_k is None:

        n_clusters = automatic_k

        k_mode = "Theo ngưỡng của main script"

    else:

        try:

            n_clusters = int(requested_k)

        except (TypeError, ValueError):

            raise ValueError("K tùy chỉnh phải là số nguyên.")

        if n_clusters < 1:

            raise ValueError("K phải ít nhất là 1.")

        if n_clusters > 30:

            raise ValueError("Bản demo giới hạn K không quá 30 để có thể trực quan đầy đủ các vòng.")

        n_clusters = min(n_clusters, sentence_count)

        k_mode = "K tùy chỉnh cho Summarizer(n_clusters=...). Main script tự chọn K=" + str(automatic_k) + "."



    # Match the production Word2Vec call and sum word vectors sentence-wise.

    model = Word2Vec(
        sentences=text_sents,
        vector_size=256,
        window=5,
        min_count=1,
        workers=1,
        sg=0,
        seed=RANDOM_SEED,
    )

    X = []

    for sent in text_sents:

        vector = np.zeros(model.vector_size)

        for word in sent:

            if word in model.wv:

                vector += model.wv[word]

        X.append(vector)

    X = np.asarray(X)



    # Trace every K-means++ restart. For each seed, compare our visible step log
    # with an actual sklearn fit from the exact same initial centers. The real
    # sklearn inertia decides the winning restart.
    template = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
    restarts = _effective_n_init(template)
    # RNG độc lập và có seed cố định: cùng input luôn duyệt cùng chuỗi khởi tạo K-means++.
    probe_rng = np.random.RandomState(RANDOM_SEED)
    all_runs_raw = []
    for restart_index in range(restarts):
        # Preserve the public sklearn initializer as the authoritative result;
        # independently replay it to capture each D²-weighted local trial.
        state_before_init = probe_rng.get_state()
        actual_rng = np.random.RandomState()
        actual_rng.set_state(state_before_init)
        initial_centers, initial_indices = kmeans_plusplus(X, n_clusters, random_state=actual_rng)
        trace_rng = np.random.RandomState()
        trace_rng.set_state(state_before_init)
        traced_seed_centers, traced_seed_indices, seed_trace = _kmeans_plusplus_trace(X, n_clusters, trace_rng)
        seed_trace_matches = bool(
            np.array_equal(initial_indices, traced_seed_indices)
            and np.allclose(initial_centers, traced_seed_centers, atol=1e-10, rtol=1e-10)
        )
        # Advance to exactly the state reached by the official function, so
        # multiple n_init runs use the same random stream as before this demo.
        probe_rng.set_state(actual_rng.get_state())
        if not seed_trace_matches:
            seed_trace = [{
                "center_index": i, "selected_index": int(index), "first": i == 0,
                "n_local_trials": 0, "probabilities": [], "trials": [],
                "trace_matches_sklearn": False,
            } for i, index in enumerate(initial_indices)]
        tol = float(np.mean(np.var(X, axis=0)) * template.tol)
        trace = _lloyd_trace(X, initial_centers, template.max_iter, tol)
        fitted = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=1,
                        max_iter=template.max_iter, tol=template.tol,
                        algorithm=template.algorithm).fit(X)
        trace_matches = bool(
            np.array_equal(trace[2], fitted.labels_)
            and np.allclose(trace[1], fitted.cluster_centers_, atol=max(tol, 1e-8))
        )
        all_runs_raw.append({
            "seed_indices": initial_indices.copy(),
            "seed_trace": seed_trace,
            "seed_trace_matches": seed_trace_matches,
            "history": trace[0],
            "centers": trace[1],
            "labels": trace[2],
            "trace_inertia": trace[3],
            "inertia": float(fitted.inertia_),
            "trace_matches": trace_matches,
            "model": fitted,
        })

    best_run_index = min(range(len(all_runs_raw)), key=lambda i: all_runs_raw[i]["inertia"])
    best_run = all_runs_raw[best_run_index]

    # In báo cáo hội tụ ra terminal mỗi khi người dùng chạy pipeline.
    # Mỗi lần khởi tạo được in riêng vì KMeans so sánh các lần chạy bằng inertia.
    print("\n" + "=" * 76)
    print("[K-MEANS] BÁO CÁO ĐIỀU KIỆN DỪNG")
    print(f"Số cụm K={n_clusters} | số lần khởi tạo={len(all_runs_raw)} | "
          f"tol={template.tol:g} | max_iter={template.max_iter}")
    print(f"Ngưỡng dịch tâm² thực tế: tolerance={tol:.10g}")
    for run_index, run_data in enumerate(all_runs_raw):
        run_history = run_data.get("history") or []
        last_frame = run_history[-1] if run_history else {}
        iterations_done = int(getattr(run_data.get("model"), "n_iter_", max(0, len(run_history) - 1)))
        if last_frame.get("converged"):
            stop_status = "ĐÃ HỘI TỤ"
            stop_reason = last_frame.get("convergence_reason") or "Đã thỏa điều kiện hội tụ của Lloyd."
        else:
            stop_status = "DỪNG DO GIỚI HẠN VÒNG LẶP"
            last_check = last_frame.get("convergence_reason") or "Không có thông tin điều kiện cuối."
            stop_reason = (f"Đạt max_iter={template.max_iter} khi chưa hội tụ. "
                           f"Kiểm tra ở vòng cuối: {last_check}")
        best_marker = " [LẦN ĐƯỢC CHỌN]" if run_index == best_run_index else ""
        print(f"\nLần khởi tạo {run_index + 1}{best_marker}: {stop_status}")
        print(f"  Số vòng Lloyd: {iterations_done}")
        print(f"  Inertia: {run_data['inertia']:.10g}")
        print(f"  Lý do dừng: {stop_reason}")
        print(f"  Trace khớp với kết quả scikit-learn: {run_data.get('trace_matches', False)}")
    print("=" * 76 + "\n")

    seed_indices = best_run["seed_indices"]
    history = best_run["history"]
    traced_centers = best_run["centers"]
    traced_labels = best_run["labels"]
    traced_inertia = best_run["trace_inertia"]
    kmeans = best_run["model"]

    closest, closest_distances = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    average_positions = []
    for cluster_index in range(n_clusters):
        member_indices = np.where(kmeans.labels_ == cluster_index)[0]
        average_positions.append(float(np.mean(member_indices)) if len(member_indices) else float("nan"))
    ordered_clusters = sorted(range(n_clusters), key=lambda index: average_positions[index])
    index_currents = [int(closest[index]) for index in ordered_clusters]
    position_index = Summarizer.find_position(index_currents, boundaries)

    # Store a complete per-cluster representative-selection trace. The repository
    # chooses the nearest sentence to each fitted center across X (not a filtered
    # within-cluster subset), so the visualization reproduces that exact behavior.
    representative_steps = []
    for order_index, cluster_index in enumerate(ordered_clusters):
        center = kmeans.cluster_centers_[cluster_index]
        delta = X - center
        distances_sq = np.einsum("nd,nd->n", delta, delta)
        chosen_index = int(closest[cluster_index])
        all_candidates = []
        for sentence_index in range(sentence_count):
            d2 = float(distances_sq[sentence_index])
            all_candidates.append({
                "index": int(sentence_index),
                "id": "C" + str(sentence_index + 1).zfill(2),
                "text": org_sents[sentence_index],
                "assigned_cluster": int(kmeans.labels_[sentence_index]),
                "distance_sq": _json_number(d2),
                "distance": _json_number(float(np.sqrt(max(d2, 0.0)))),
                "is_representative": sentence_index == chosen_index,
            })
        all_candidates.sort(key=lambda row: (float(row["distance_sq"]), row["index"]))
        chosen_d2 = float(distances_sq[chosen_index])
        representative_steps.append({
            "cluster": int(cluster_index),
            "order": int(order_index),
            "center_preview": [_json_number(v) for v in center[:8]],
            "center_vector": [_json_number(v) for v in center],
            "selected_index": chosen_index,
            "selected_id": "C" + str(chosen_index + 1).zfill(2),
            "selected_text": org_sents[chosen_index],
            "selected_distance_sq": _json_number(chosen_d2),
            "selected_distance": _json_number(float(np.sqrt(max(chosen_d2, 0.0)))),
            "selected_in_cluster": bool(int(kmeans.labels_[chosen_index]) == int(cluster_index)),
            "candidates": all_candidates,
        })

    # PCA is only for drawing; assignment, distance, and centroid updates stay
    # in the original 256-dimensional Word2Vec space.
    pca = PCA(n_components=2, svd_solver="full")
    pca_sentences = pca.fit_transform(X)

    total_trace_frames = sum(len(run["history"]) for run in all_runs_raw)
    total_distance_values = total_trace_frames * sentence_count * n_clusters
    total_center_values = total_trace_frames * n_clusters * model.vector_size * 2
    full_distances_all_runs = total_distance_values <= 4_000_000
    full_vectors_all_runs = total_center_values <= 3_000_000

    def serialize_kmeans_history(trace, include_full_distances=True, include_full_vectors=True):
        serialized = []
        for step_index, frame in enumerate(trace):
            frame_centers = pca.transform(frame["centers"])
            frame_centers_after = pca.transform(frame.get("centers_after", frame["centers"]))
            cluster_updates_json = []
            for update in frame.get("cluster_updates", []):
                item = {
                    "cluster": int(update["cluster"]),
                    "members": [int(i) for i in update.get("members", [])],
                    "count": int(update["count"]),
                    "old_preview": [_json_number(v) for v in update.get("old_preview", [])],
                    "new_preview": [_json_number(v) for v in update.get("new_preview", [])],
                    "shift_sq": _json_number(update.get("shift_sq")),
                    "relocated": update.get("relocated", []),
                }
                if include_full_vectors:
                    item["old_vector"] = [_json_number(v) for v in update.get("old_vector", [])]
                    item["new_vector"] = [_json_number(v) for v in update.get("new_vector", [])]
                cluster_updates_json.append(item)
            serialized.append({
                "iteration": int(frame.get("iteration", step_index)),
                "label": frame["label"], "note": frame["note"],
                "centers": [{"x": _json_number(c[0]), "y": _json_number(c[1])} for c in frame_centers],
                "centers_after": [{"x": _json_number(c[0]), "y": _json_number(c[1])} for c in frame_centers_after],
                "assignments": [int(x) for x in frame["assignments"]],
                "distance_matrix": [[_json_number(v) for v in row] for row in frame.get("distance_matrix", [])] if include_full_distances else [],
                "has_full_distances": bool(include_full_distances),
                "distances": [_json_number(x) for x in frame["distances"]],
                "post_assignments": [int(x) for x in frame.get("post_assignments", frame["assignments"])],
                "post_distances": [_json_number(x) for x in frame.get("post_distances", frame["distances"])],
                "inertia": _json_number(frame["inertia"]),
                "shift": _json_number(frame.get("shift")),
                "cluster_updates": cluster_updates_json,
                "converged": bool(frame.get("converged", False)),
                "convergence_reason": frame.get("convergence_reason", ""),
            })
        return serialized

    restarts_json = []
    for run_index, run_data in enumerate(all_runs_raw):
        include_distances = full_distances_all_runs or run_index == best_run_index
        include_vectors = full_vectors_all_runs or run_index == best_run_index
        restarts_json.append({
            "run": run_index + 1,
            "seed_indices": [int(x) for x in run_data["seed_indices"]],
            "seed_trace": run_data.get("seed_trace", []),
            "seed_trace_matches": bool(run_data.get("seed_trace_matches", False)),
            "final_inertia": _json_number(run_data["inertia"]),
            "trace_inertia": _json_number(run_data["trace_inertia"]),
            "trace_matches": bool(run_data["trace_matches"]),
            "best": run_index == best_run_index,
            "has_full_distances": bool(include_distances),
            "has_full_vectors": bool(include_vectors),
            "history": serialize_kmeans_history(run_data["history"], include_distances, include_vectors),
        })
    history_json = restarts_json[best_run_index]["history"]

    public_sentences = []

    for i, sentence in enumerate(sentences):

        public_sentences.append({

            "index": i, "id": "C" + str(i + 1).zfill(2),

            "text": sentence.getOGwords(), "tokens": sentence.getProcessedWords(),

            "doc": source_doc[i], "doc_label": "Tài liệu " + str(source_doc[i] + 1),
            "input_position": source_positions[i] + 1,

            "x": _json_number(pca_sentences[i][0]), "y": _json_number(pca_sentences[i][1]),
            "vector_preview": [_json_number(v) for v in X[i][:8]],
            "vector": [_json_number(v) for v in X[i]],
            "vector_norm": _json_number(float(np.linalg.norm(X[i]))),
            "vector_norm_sq": _json_number(float(np.dot(X[i], X[i]))),

        })

    source_ids = {i: "C" + str(i + 1).zfill(2) for i in range(sentence_count)}

    representative_by_index = {}

    for cluster_index in ordered_clusters:

        rep_index = int(closest[cluster_index])

        members = [int(i) for i in np.where(kmeans.labels_ == cluster_index)[0]]

        representative_by_index[rep_index] = {

            "index": rep_index, "id": source_ids[rep_index],

            "text": org_sents[rep_index], "cluster": int(cluster_index),

            "members": [source_ids[i] for i in members],

        }

    # Summarizer.kmean_summarizer returns the same order through find_position.

    new_sents = [org_sents[index] for index in position_index]

    object_sents = [sentences[index] for index in position_index]

    representatives = [representative_by_index[index] for index in position_index]



    # Run and inspect the repository's own CentroidBow implementation.
    centroid = CentroidBow()
    centroid_selected = centroid.sumarize(
        org_sents_for_reps := [item.getOGwords() for item in object_sents], new_sents, 20
    )
    vectorizer = CountVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(new_sents)
    transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
    tfidf = transformer.fit_transform(matrix).toarray()
    vocab = vectorizer.get_feature_names_out().tolist()
    count_dense = matrix.toarray()
    df_values = np.asarray(matrix.getnnz(axis=0)).ravel()
    idf_values = np.asarray(transformer.idf_)

    centroid_raw = tfidf.sum(axis=0)
    centroid_max = float(centroid_raw.max()) if centroid_raw.size else 0.0
    centroid_normalized = centroid_raw / centroid_max if centroid_max else np.zeros_like(centroid_raw)
    centroid_threshold = float(centroid.topic_threshold)
    centroid_vector = centroid_normalized.copy()
    centroid_vector[centroid_vector <= centroid_threshold] = 0
    sim_threshold = float(CentroidBow().sim_threshold)

    sorted_centroid = []
    cosine_details = []
    centroid_vector_norm = float(np.linalg.norm(centroid_vector))
    for i in range(tfidf.shape[0]):
        score = CentroidBow.similarity(tfidf[i, :], centroid_vector)
        sentence_tfidf = tfidf[i, :]
        sentence_norm = float(np.linalg.norm(sentence_tfidf))
        dot_product = float(np.dot(sentence_tfidf, centroid_vector))
        denominator = sentence_norm * centroid_vector_norm
        cosine_score = dot_product / denominator if denominator > 0 else 0.0
        cosine_details.append({
            "score": float(cosine_score), "dot": dot_product,
            "sentence_norm": sentence_norm, "centroid_norm": centroid_vector_norm,
        })
        sorted_centroid.append((i, float(score)))
    sorted_centroid.sort(key=lambda item: item[1], reverse=True)

    centroid_similarity_source = ""
    try:
        import inspect
        centroid_similarity_source = inspect.getsource(CentroidBow.similarity)
    except (OSError, TypeError):
        centroid_similarity_source = "Không thể đọc source bằng inspect.getsource; điểm vẫn được tính trực tiếp bằng CentroidBow.similarity()."

    chosen_centroid_indices = []
    centroid_decisions = {}
    centroid_process = []
    for rank, (candidate_index, score) in enumerate(sorted_centroid, start=1):
        comparisons = []
        include = True
        reason = "Không có similarity nào vượt ngưỡng với câu đã giữ trước đó."
        processed_for_redundancy = len(chosen_centroid_indices) < 20
        if not processed_for_redundancy:
            include = False
            reason = "Đã đạt giới hạn 20 câu của CentroidBow; ứng viên này không còn được đưa vào vòng lọc trùng."
        else:
            for previously_selected in chosen_centroid_indices:
                similarity = float(CentroidBow().similarity(tfidf[candidate_index, :], tfidf[previously_selected, :]))
                comparisons.append({
                    "against_index": int(previously_selected),
                    "against_id": "R" + str(previously_selected + 1).zfill(2),
                    "against_text": new_sents[previously_selected],
                    "similarity": _json_number(similarity),
                    "threshold": _json_number(sim_threshold),
                    "rejected": bool(similarity > sim_threshold),
                })
                if similarity > sim_threshold:
                    include = False
                    reason = "Bị loại vì similarity với ít nhất một câu đã giữ lớn hơn ngưỡng " + format(sim_threshold, ".4g") + "."
        if include and processed_for_redundancy:
            chosen_centroid_indices.append(candidate_index)
            reason = "Giữ: không có so sánh nào vượt ngưỡng và chưa đạt giới hạn số câu."

        maximum_redundancy = max((float(x["similarity"]) for x in comparisons), default=None)
        centroid_decisions[candidate_index] = {
            "score": _json_number(score),
            "selected": include,
            "redundancy": _json_number(maximum_redundancy),
            "comparisons": comparisons,
            "reason": reason,
            "rank": rank,
        }

        terms = []
        for j in np.flatnonzero(count_dense[candidate_index]):
            terms.append({
                "term": vocab[int(j)],
                "tf": int(count_dense[candidate_index, j]),
                "df": int(df_values[j]),
                "idf": _json_number(idf_values[j]),
                "tfidf": _json_number(tfidf[candidate_index, j]),
                "centroid_sum": _json_number(centroid_raw[j]),
                "centroid_normalized": _json_number(centroid_normalized[j]),
                "centroid_after_threshold": _json_number(centroid_vector[j]),
            })
        cosine_info = cosine_details[candidate_index]
        cosine_products = tfidf[candidate_index, :] * centroid_vector
        cosine_term_indices = np.flatnonzero(cosine_products > 0)
        cosine_term_indices = sorted(cosine_term_indices, key=lambda j: float(cosine_products[j]), reverse=True)[:20]
        cosine_terms = [{
            "term": vocab[int(j)], "tfidf": _json_number(tfidf[candidate_index, j]),
            "centroid_weight": _json_number(centroid_vector[j]),
            "product": _json_number(cosine_products[j]),
        } for j in cosine_term_indices]
        centroid_process.append({
            "index": int(candidate_index),
            "id": "R" + str(candidate_index + 1).zfill(2),
            "text": new_sents[candidate_index],
            "rank": int(rank),
            "score": _json_number(score),
            "cosine_score": _json_number(cosine_info["score"]),
            "cosine_dot": _json_number(cosine_info["dot"]),
            "cosine_sentence_norm": _json_number(cosine_info["sentence_norm"]),
            "cosine_centroid_norm": _json_number(cosine_info["centroid_norm"]),
            "cosine_terms": cosine_terms,
            "terms": terms,
            "comparisons": comparisons,
            "manual_selected": bool(include),
            "selected": bool(include),
            "reason": reason,
            "redundancy": _json_number(maximum_redundancy),
            "processed_for_redundancy": bool(processed_for_redundancy),
        })

    # CentroidBow returns sentence text rather than indices. Match each selected
    # text to its first source occurrence, consistent with position restoration.
    selected_by_repo = []
    for text in centroid_selected:
        for index, raw in enumerate(org_sents_for_reps):
            if text == raw:
                selected_by_repo.append(index)
                break
    repository_selected_indices = set(selected_by_repo)
    index_sents = list(selected_by_repo)
    cosine_reasons = {}
    cosine_comparisons = {}
    cosine_redundancy = {}

    if centroid_mode == "cosine":
        # Alternate, explicit topic-selection mode requested for visual study:
        # cosine(sentence TF-IDF, filtered centroid) >= threshold, then redundancy filtering.
        cosine_order = sorted(range(len(new_sents)), key=lambda i: cosine_details[i]["score"], reverse=True)
        index_sents = []
        for candidate_index in cosine_order:
            cosine_value = cosine_details[candidate_index]["score"]
            if cosine_value < cosine_threshold:
                cosine_reasons[candidate_index] = (
                    "Bỏ vì cosine " + format(cosine_value, ".4f") + " < ngưỡng " + format(cosine_threshold, ".4f") + "."
                )
                continue
            if len(index_sents) >= 20:
                cosine_reasons[candidate_index] = "Bỏ vì đã đủ giới hạn 20 câu."
                continue
            comparisons_for_candidate = []
            comparison_scores = []
            duplicate = False
            for previously_selected in index_sents:
                pair_score = float(CentroidBow().similarity(tfidf[candidate_index, :], tfidf[previously_selected, :]))
                comparison_scores.append(pair_score)
                comparisons_for_candidate.append({
                    "against_index": int(previously_selected),
                    "against_id": "R" + str(previously_selected + 1).zfill(2),
                    "against_text": new_sents[previously_selected],
                    "similarity": _json_number(pair_score),
                    "threshold": _json_number(sim_threshold),
                    "rejected": bool(pair_score > sim_threshold),
                })
                if pair_score > sim_threshold:
                    duplicate = True
                    break
            cosine_comparisons[candidate_index] = comparisons_for_candidate
            cosine_redundancy[candidate_index] = max(comparison_scores, default=None)
            if duplicate:
                cosine_reasons[candidate_index] = (
                    "Đạt ngưỡng cosine nhưng bị loại vì similarity với câu đã giữ vượt ngưỡng lọc trùng "
                    + format(sim_threshold, ".4f") + "."
                )
            else:
                index_sents.append(candidate_index)
                cosine_reasons[candidate_index] = "Giữ: cosine đạt ngưỡng và không trùng quá ngưỡng với câu đã giữ."
        if not index_sents:
            raise ValueError(
                "Không có câu nào đạt ngưỡng cosine " + format(cosine_threshold, ".3f") +
                ". Hãy giảm ngưỡng cosine hoặc chuyển sang CentroidBow gốc."
            )

    active_selected_indices = set(index_sents)
    for item in centroid_process:
        candidate_index = item["index"]
        item["repository_selected"] = candidate_index in repository_selected_indices
        item["selected"] = candidate_index in active_selected_indices
        if centroid_mode == "cosine":
            item["reason"] = cosine_reasons.get(candidate_index, "Không được chọn.")
            item["comparisons"] = cosine_comparisons.get(candidate_index, [])
            item["redundancy"] = _json_number(cosine_redundancy.get(candidate_index))
            item["processed_for_redundancy"] = candidate_index in cosine_comparisons
            item["actual_decision_note"] = (
                "Tập đưa vào MMR đang dùng bộ lọc cosine + ngưỡng; trường đối chiếu cho biết CentroidBow gốc có giữ câu này hay không."
            )
        elif item["selected"] != item["manual_selected"]:
            item["actual_decision_note"] = (
                "Trace greedy theo index khác đầu ra thật của CentroidBow.sumarize(); có thể do câu trùng nội dung/ánh xạ chuỗi. "
                "Đầu ra thực tế từ repository được ưu tiên."
            )
        else:
            item["actual_decision_note"] = "Trace greedy theo index khớp với đầu ra thực tế của CentroidBow.sumarize()."

    ranking = []
    for candidate_index, score in sorted_centroid:
        decision = centroid_decisions.get(candidate_index, {})
        active_selected = candidate_index in active_selected_indices
        repository_selected = candidate_index in repository_selected_indices
        active_reason = cosine_reasons.get(candidate_index, decision.get("reason", "Không có trong kết quả."))
        ranking.append({
            "index": candidate_index,
            "id": "R" + str(candidate_index + 1).zfill(2),
            "text": new_sents[candidate_index],
            "score": _json_number(score),
            "cosine_score": _json_number(cosine_details[candidate_index]["score"]),
            "selected": active_selected,
            "repository_selected": repository_selected,
            "decision": "Giữ" if active_selected else "Bỏ: " + active_reason,
            "redundancy": decision.get("redundancy"),
        })

    all_centroid_terms = [{
        "term": vocab[j],
        "raw_sum": _json_number(centroid_raw[j]),
        "normalized": _json_number(centroid_normalized[j]),
        "after_threshold": _json_number(centroid_vector[j]),
    } for j in range(len(vocab))]

    new_sents_mmr = [object_sents[item] for item in index_sents]

    if not new_sents_mmr:

        raise ValueError("CentroidBow trả về 0 ứng viên; main pipeline sẽ không có câu Q để bắt đầu MMR. Hãy đổi văn bản hoặc K.")

    idf = MMR.IDFs(new_sents_mmr)

    query = new_sents_mmr[0]

    remaining = new_sents_mmr[1:].copy()

    mmr_pool = new_sents_mmr.copy()

    summary = [query]

    target = mmr_target_count(len(new_sents_mmr))

    sum_len = 0

    rounds = []

    mmr_engine = MMR()

    while sum_len < target - 1:

        if not remaining:

            raise ValueError("MMR.makeSummary trong repository gọi max() trên tập rỗng ở chế độ này; cần ít nhất một ứng viên ngoài Q.")

        score_map = {}

        candidate_details = []

        selected_before = list(summary)
        remaining_before = list(remaining)

        for candidate in remaining:

            rel = MMR.sentenceSim(candidate, query, idf)

            red_values = [MMR.sentenceSim(candidate, selected, idf) for selected in summary]

            red = max([-float("inf")] + red_values)

            score = mmr_engine.MMRScore(candidate, query, summary, lambta, idf)

            score_map[candidate] = score

            red_details = []

            for selected_index, selected in enumerate(summary):

                detail = _sentence_similarity_detail(candidate, selected, idf)

                detail["against_id"] = source_ids.get(sentences.index(selected), "Q")

                pair_score = red_values[selected_index]
                detail["is_max"] = bool(
                    pair_score == red
                    or (
                        math.isfinite(float(pair_score))
                        and math.isfinite(float(red))
                        and math.isclose(float(pair_score), float(red), rel_tol=1e-9, abs_tol=1e-10)
                    )
                )
                red_details.append(detail)

            rel_detail = _sentence_similarity_detail(candidate, query, idf)

            rel_detail["against_id"] = source_ids.get(sentences.index(query), "Q")

            # Show the textbook MMR arithmetic separately from the repository's
            # own MMRScore result. This detects argument-order or implementation
            # differences instead of silently presenting inconsistent values.
            formula_score = lambta * rel - (1.0 - lambta) * red

            candidate_details.append({

                "sentence": candidate,

                "id": source_ids.get(sentences.index(candidate), "?"),

                "rel": _json_number(rel), "red": _json_number(red),

                "score": _json_number(score),
                "formula_score": _json_number(formula_score),
                "score_matches_formula": _same_number(score, formula_score),

                "rel_detail": rel_detail, "red_details": red_details,

            })

        winner = max(score_map, key=score_map.get)

        winner_detail = next(item for item in candidate_details if item["sentence"] is winner)

        winner_detail["winner"] = True

        for item in candidate_details:

            item["winner"] = item["sentence"] is winner

        rounds.append({
            "selected_before": [{"id": source_ids.get(sentences.index(x), "?"), "text": x.getOGwords()} for x in selected_before],
            "selected_after": [{"id": source_ids.get(sentences.index(x), "?"), "text": x.getOGwords()} for x in selected_before + [winner]],
            "remaining_before": [{"id": source_ids.get(sentences.index(x), "?"), "text": x.getOGwords()} for x in remaining_before],
            "remaining_after": [{"id": source_ids.get(sentences.index(x), "?"), "text": x.getOGwords()} for x in remaining_before if x is not winner],
            "candidates": [{key: value for key, value in item.items() if key != "sentence"} for item in candidate_details],
            "winner_id": source_ids.get(sentences.index(winner), "?"),
        })

        summary.append(winner)

        remaining.remove(winner)

        if len(remaining) == 0:

            break

        sum_len += 1



    mmr_strings = [item.getOGwords() for item in summary]

    requested_length = payload.get("summary_length")
    if requested_length is not None and (isinstance(requested_length, bool) or not isinstance(requested_length, int) or not 1 <= requested_length <= 16):
        raise ValueError("Số câu phải là số nguyên từ 1 đến 16, hoặc để trống để tự động.")
    if requested_length is None:
        requested_length = len(mmr_strings)
    final_target = min(requested_length, n_clusters, len(new_sents), len(mmr_strings))
    mmr_for_final = mmr_strings[:final_target]

    final_index = Summarizer(n_clusters=n_clusters, len_summary=final_target).add_position(

        new_sents, mmr_for_final, final_target, index_sents, n_clusters

    )

    final = []

    for index in final_index:

        source_index = position_index[index]

        final.append({

            "id": source_ids[source_index], "text": new_sents[index],

            "doc_label": "Tài liệu " + str(source_doc[source_index] + 1),

            "source_index": source_index,

        })



    for frame in history_json:

        frame["assignments"] = [int(x) for x in frame["assignments"]]

    return {

        "documents": len(documents), "boundaries": boundaries,

        "dropped_count": dropped_count, "sentences": public_sentences,

        "colors": list(range(n_clusters)),

        "kmeans": {"k": n_clusters, "automatic_k": automatic_k, "k_mode": k_mode,
                   "n_init": restarts, "history": history_json,
                   "restarts": restarts_json, "best_run": best_run_index + 1,
                   "trace_matches_sklearn": bool(best_run["trace_matches"]),
                   "seed_indices": [int(x) for x in seed_indices],
                   "distance_detail_limited": not full_distances_all_runs,
                   "vector_detail_limited": not full_vectors_all_runs},

        "representatives": representatives,
        "representative_steps": representative_steps,

        "centroid": {"selected": [{"id": "R" + str(index + 1).zfill(2), "text": new_sents[index]} for index in index_sents],
                     "selection_mode": centroid_mode, "cosine_threshold": _json_number(cosine_threshold),
                     "repository_selected": [{"id": "R" + str(index + 1).zfill(2), "text": new_sents[index]} for index in sorted(repository_selected_indices)],
                     "ranking": ranking,
                     "process": [{
                         "index": int(item["index"]), "id": item["id"], "text": item["text"],
                         "rank": int(item["rank"]), "score": item["score"],
                         "cosine_score": item["cosine_score"], "cosine_dot": item["cosine_dot"],
                         "cosine_sentence_norm": item["cosine_sentence_norm"],
                         "cosine_centroid_norm": item["cosine_centroid_norm"],
                         "cosine_terms": item["cosine_terms"], "terms": item["terms"],
                         "comparisons": item["comparisons"], "manual_selected": item["manual_selected"],
                         "repository_selected": item.get("repository_selected", item["selected"]),
                         "selected": item["selected"], "reason": item["reason"],
                         "redundancy": item["redundancy"],
                         "processed_for_redundancy": item["processed_for_redundancy"],
                         "actual_decision_note": item["actual_decision_note"],
                     } for item in centroid_process],
                     "all_centroid_terms": all_centroid_terms,
                     "threshold": _json_number(sim_threshold),
                     "threshold_topic": _json_number(centroid_threshold),
                     "similarity_source": centroid_similarity_source},

        "mmr": {"pool": [{"id": source_ids.get(sentences.index(x), "?"), "text": x.getOGwords()} for x in mmr_pool],

                "selected": [{"id": source_ids.get(sentences.index(x), "?"), "text": x.getOGwords()} for x in summary],

                "rounds": rounds, "lambda": lambta, "mode": mode, "target": target,

                "idf": {key: _json_number(value) for key, value in idf.items()}},

        "final": final, "final_target": final_target,

    }





class Handler(BaseHTTPRequestHandler):

    server_version = "PipelineDemo/1.0"



    def do_GET(self):

        if self.path not in ("/", "/index.html"):

            self.send_error(404)

            return

        body = PAGE.encode("utf-8")

        self.send_response(200)

        self.send_header("Content-Type", "text/html; charset=utf-8")

        self.send_header("Content-Length", str(len(body)))

        self.send_header("Cache-Control", "no-store")

        self.end_headers()

        self.wfile.write(body)



    def do_POST(self):

        if self.path != "/api/run":

            self.send_error(404)

            return

        try:

            length = int(self.headers.get("Content-Length", "0"))

            if length > 1_000_000:

                raise ValueError("Yêu cầu tối đa 1 MB.")

            payload = json.loads(self.rfile.read(length).decode("utf-8"))

            result = run_pipeline(payload)

            body = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")

            self.send_response(200)

        except Exception as error:

            body = json.dumps({"error": str(error)}, ensure_ascii=False).encode("utf-8")

            self.send_response(400)

        self.send_header("Content-Type", "application/json; charset=utf-8")

        self.send_header("Content-Length", str(len(body)))

        self.send_header("Cache-Control", "no-store")

        self.end_headers()

        self.wfile.write(body)



    def log_message(self, format_string, *args):

        print("[%s] %s" % (self.log_date_time_string(), format_string % args))





def main():

    parser = argparse.ArgumentParser(description="Interactive local visualization of the summary pipeline")

    parser.add_argument("--host", default="127.0.0.1", help="Bind address; default is local-only")

    parser.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), Handler)

    print("Pipeline demo: http://127.0.0.1:%d" % args.port)

    print("Press Ctrl+C to stop.")

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print("\nServer stopped.")

    finally:

        server.server_close()





if __name__ == "__main__":

    main()
