#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1.py — Windows 单文件版：读取原始表 → 构建分析数据 → 统一输出到一个 TXT 文本报告
用途：论文附录·文件一（整合版；无图版）
- 读取你的 Excel（默认使用你给定的 Windows 路径）
- 构造论文所需变量（HighArousal/Positive/Negative/TitleLen/HasQ/HasExcl/ln_fans 等）
- 检查并裁剪四个比例型因变量到 [0,1]；统一的轻量缺失处理
- 生成：变量—口径对照、描述性统计、分组均值、处理摘要 —— 全部写入一个 TXT
- 同步保存清洗后的分析数据 CSV（供后续回归使用）；不保存 parquet（避免环境依赖）

最小用法（双击或直接运行都可；默认路径已写入）：
  python P1.py

依赖：
  pip install pandas openpyxl
"""

from __future__ import annotations
import argparse, hashlib, json, math, os, sys, datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ===================== 默认 Windows 路径（可用命令行覆盖） =====================
DEFAULT_INPUT_PATH = r""
DEFAULT_OUTFILE    = r""
DEFAULT_OUTDIR     = r""   # 用于保存清洗后的 CSV 和日志


# ===================== 必需字段（硬校验） =====================
REQUIRED_COLUMNS = [
    "title_y",
    "category_y",
    "duration_y",
    "num_UP_fans_y",
    "emotion",          # 主导情绪（类别）
    "emotion_value",    # 情绪值（连续，约 [-1,1]）
    "ratio_like_y",
    "ratio_coin_y",
    "ratio_fav_y",
    "ratio_danmaku_y",
]

# 主导情绪 → 唤起强度映射（固定口径；未知类别不猜测）
EMOTION_TO_AROUSAL: Dict[str, int] = {
    # 高唤起（1）
    "excitement": 1, "anger": 1, "fear": 1, "disgust": 1, "awe": 1, "amusement": 1, "surprise": 1,
    # 低唤起（0）
    "contentment": 0, "sadness": 0, "calm": 0, "relief": 0,
}


# ===================== 工具函数 =====================
def sha256_of_file(path: str) -> str:
    if not os.path.exists(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()

def ensure_dirs(base: str) -> Dict[str, str]:
    paths = {
        "clean": os.path.join(base, "data", "clean"),
        "logs": os.path.join(base, "logs"),
    }
    for p in paths.values(): os.makedirs(p, exist_ok=True)
    return paths

def clip_01(series: pd.Series) -> Tuple[pd.Series, int]:
    s = pd.to_numeric(series, errors="coerce")
    n = int(((s < 0) | (s > 1)).sum())
    return s.clip(0,1), n

def fmt_float(x, nd=6) -> str:
    if x is None: return "NA"
    try:
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf): return "NA"
        return f"{xf:.{nd}f}"
    except Exception:
        return "NA"

def build_dictionary_rows():
    return [
        ("ratio_like_y",   "原始字段", "ratio [0,1]",     "单位播放对应的点赞概率"),
        ("ratio_coin_y",   "原始字段", "ratio [0,1]",     "单位播放对应的投币概率"),
        ("ratio_fav_y",    "原始字段", "ratio [0,1]",     "单位播放对应的收藏概率"),
        ("ratio_danmaku_y","原始字段", "ratio [0,1]",     "单位播放触发的即时讨论强度"),
        ("emotion",        "原始字段", "category",        "主导情绪类别"),
        ("emotion_value",  "原始字段", "float ~[-1,1]",   "情绪倾向连续值，正为正向，负为负向"),
        ("HighArousal",    "派生",     "binary {0,1}",    "由主导情绪映射得到的高唤起指示"),
        ("Positive",       "派生",     "binary {0,1}",    "emotion_value>0 指示"),
        ("Negative",       "派生",     "binary {0,1}",    "emotion_value<0 指示"),
        ("TitleLen",       "派生",     "int",             "标题字符数（去首尾空白）"),
        ("HasQ",           "派生",     "binary {0,1}",    "标题是否包含问号（？ / ?）"),
        ("HasExcl",        "派生",     "binary {0,1}",    "标题是否包含感叹号（！ / !）"),
        ("num_UP_fans_y",  "原始字段", "int",             "UP 主粉丝数（原始）"),
        ("ln_fans",        "派生",     "float",           "UP 粉丝数的自然对数（对 0 用 ln(1+fans)）"),
        ("duration_y",     "原始字段", "float/int（秒）",  "视频时长（秒）"),
        ("category_y",     "原始字段", "category",        "内容分区（参照组在回归脚本中设定为众数）"),
    ]

def safe_len_title(s):
    if pd.isna(s): return np.nan
    return len(str(s).strip())

def has_q(s):
    if pd.isna(s): return 0
    t = str(s); return int(("？" in t) or ("?" in t))

def has_excl(s):
    if pd.isna(s): return 0
    t = str(s); return int(("！" in t) or ("!" in t))


# ===================== 主流程 =====================
def run(input_path: str, outfile: str, outdir: str) -> int:
    paths = ensure_dirs(outdir)

    # 1) 读取 Excel（优先使用 openpyxl；未安装则给出清晰提示）
    print(f"[INFO] Loading: {input_path}")
    if not os.path.exists(input_path):
        print(f"[ERROR] 输入的 Excel 路径不存在：{input_path}")
        return 1
    try:
        df = pd.read_excel(input_path, engine="openpyxl")
    except ImportError:
        print("[ERROR] 读取 Excel 失败：缺少 openpyxl。请先安装：pip install openpyxl")
        return 1
    except Exception as e:
        print(f"[ERROR] 读取 Excel 失败：{e}")
        return 1

    # 2) 硬校验列
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write("# 单文件报告（失败）\n")
            f.write(f"缺少必需列：{missing}\n")
        print(f"[ERROR] 原始数据缺少必需列：{missing}")
        return 2

    data = df[REQUIRED_COLUMNS].copy()

    # 3) 派生变量（严格按正文口径）
    data["TitleLen"] = data["title_y"].apply(safe_len_title)
    data["HasQ"] = data["title_y"].apply(has_q).astype(int)
    data["HasExcl"] = data["title_y"].apply(has_excl).astype(int)

    fans = pd.to_numeric(data["num_UP_fans_y"], errors="coerce")
    fans_nonpos = int((fans <= 0).sum())
    data["ln_fans"] = np.log(np.where(fans > 0, fans, 1.0))  # ln(1)=0 for nonpositive

    data["category_y"] = data["category_y"].astype(str)

    emo_cat = data["emotion"].astype(str).str.lower()
    data["HighArousal"] = emo_cat.map(EMOTION_TO_AROUSAL)
    unknown_arousal = int(data["HighArousal"].isna().sum())
    emo_val = pd.to_numeric(data["emotion_value"], errors="coerce")
    data["Positive"] = (emo_val > 0).astype(int)
    data["Negative"] = (emo_val < 0).astype(int)

    # 4) 比例变量裁剪到 [0,1] 并记录裁剪数
    clip_summary = {}
    for col in ["ratio_like_y","ratio_coin_y","ratio_fav_y","ratio_danmaku_y"]:
        data[col], nclip = clip_01(data[col])
        clip_summary[col] = int(nclip)

    # 5) 轻量缺失处理（连续：中位数；二元：填 0）
    for col in ["TitleLen", "ln_fans", "duration_y"]:
        if data[col].isna().any():
            med = float(pd.to_numeric(data[col], errors="coerce").median())
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(med)
    for col in ["HasQ","HasExcl","HighArousal","Positive","Negative"]:
        if data[col].isna().any():
            data[col] = data[col].fillna(0).astype(int)

    # 6) 描述性统计
    desc_cols = [
        "ratio_like_y","ratio_coin_y","ratio_fav_y","ratio_danmaku_y",
        "HighArousal","Positive","Negative","TitleLen","HasQ","HasExcl","ln_fans","duration_y"
    ]
    desc_rows = []
    for c in desc_cols:
        x = pd.to_numeric(data[c], errors="coerce")
        n = int(x.notna().sum())
        desc_rows.append({
            "variable": c,
            "N": n,
            "missing": int(x.isna().sum()),
            "mean": fmt_float(np.nanmean(x) if n>0 else np.nan),
            "std": fmt_float(np.nanstd(x, ddof=1) if n>1 else np.nan),
            "median": fmt_float(np.nanmedian(x) if n>0 else np.nan),
            "p10": fmt_float(np.nanpercentile(x,10) if n>0 else np.nan),
            "p90": fmt_float(np.nanpercentile(x,90) if n>0 else np.nan),
        })
    desc_df = pd.DataFrame(desc_rows)

    # 7) 分组均值（四个因变量）—— 用 pandas 赋值，避免 NumPy 混合 dtype 报错
    targets = ["ratio_like_y","ratio_coin_y","ratio_fav_y","ratio_danmaku_y"]

    def group_table(df_: pd.DataFrame, gcol: str, allowed=None) -> pd.DataFrame:
        sub = df_.copy()
        if allowed is not None:
            sub = sub[sub[gcol].isin(allowed)]
        rows = []
        # 不使用 groupby(..., dropna=False) 以兼容旧版 pandas
        for g, gdf in sub.groupby(gcol):
            row = {"group": str(g), "n": int(gdf.shape[0])}
            for t in targets:
                gx = pd.to_numeric(gdf[t], errors="coerce")
                row[t] = fmt_float(gx.mean(), nd=6)
            rows.append(row)
        return pd.DataFrame(rows)

    tmp = data.copy()
    # 关键修复：避免 np.where 把 "Positive"/"Negative" 与 NaN 混在同一个 NumPy 数组里
    tmp["Sentiment2"] = None  # dtype=object
    tmp.loc[tmp["Positive"] == 1, "Sentiment2"] = "Positive"
    tmp.loc[tmp["Negative"] == 1, "Sentiment2"] = "Negative"

    g_high = group_table(data, "HighArousal")                       # 已经是 0/1，无缺失
    g_sent = group_table(tmp.dropna(subset=["Sentiment2"]), "Sentiment2", allowed=["Positive","Negative"])
    g_q    = group_table(data, "HasQ")
    g_excl = group_table(data, "HasExcl")

    # 8) 保存清洗数据（CSV；不保存 parquet）
    clean_csv = os.path.join(paths["clean"], "analytic_dataset.csv")
    data.to_csv(clean_csv, index=False, encoding="utf-8-sig")

    # 9) 写统一 TXT 报告
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("# 附录统一报告（数据构建与描述性统计）\n")
        f.write(f"生成时间：{now}\n\n")

        f.write("## 输入文件\n")
        f.write(f"- 路径：{input_path}\n")
        f.write(f"- SHA256：{sha256_of_file(input_path)}\n")
        f.write(f"- 行数（读取后）：{data.shape[0]}\n\n")

        # 变量字典
        f.write("## 变量—字段—口径对照表\n")
        dict_rows = build_dictionary_rows()
        header = ["名称","来源","类型","定义"]
        colw = [20,12,18,50]
        f.write("".ljust(sum(colw)+6,"-")+"\n")
        f.write(f"{header[0]:<{colw[0]}} | {header[1]:<{colw[1]}} | {header[2]:<{colw[2]}} | {header[3]:<{colw[3]}}\n")
        f.write("".ljust(sum(colw)+6,"-")+"\n")
        for n,src,typ,defi in dict_rows:
            f.write(f"{n:<{colw[0]}} | {src:<{colw[1]}} | {typ:<{colw[2]}} | {defi:<{colw[3]}}\n")
        f.write("".ljust(sum(colw)+6,"-")+"\n\n")

        # 描述性统计
        f.write("## 描述性统计（Table 1）\n")
        header = ["variable","N","missing","mean","std","median","p10","p90"]
        colw = [20,8,8,14,14,14,12,12]
        f.write("".ljust(sum(colw)+7,"-")+"\n")
        f.write(" | ".join([f"{h:<{w}}" for h,w in zip(header,colw)])+"\n")
        f.write("".ljust(sum(colw)+7,"-")+"\n")
        for _,r in desc_df.iterrows():
            vals = [r["variable"], r["N"], r["missing"], r["mean"], r["std"], r["median"], r["p10"], r["p90"]]
            f.write(" | ".join([f"{str(v):<{w}}" for v,w in zip(vals,colw)])+"\n")
        f.write("".ljust(sum(colw)+7,"-")+"\n\n")

        # 分组均值
        def dump_group(df_, title):
            f.write(f"## 分组均值：{title}\n")
            header = ["group","n"] + targets
            colw = [10,8,16,16,16,16]
            f.write("".ljust(sum(colw)+7,"-")+"\n")
            f.write(" | ".join([f"{h:<{w}}" for h,w in zip(header,colw)])+"\n")
            f.write("".ljust(sum(colw)+7,"-")+"\n")
            for _,r in df_.iterrows():
                row = [r.get("group",""), r.get("n",""),
                       r.get("ratio_like_y",""), r.get("ratio_coin_y",""),
                       r.get("ratio_fav_y",""), r.get("ratio_danmaku_y","")]
                f.write(" | ".join([f"{str(v):<{w}}" for v,w in zip(row,colw)])+"\n")
            f.write("".ljust(sum(colw)+7,"-")+"\n\n")

        dump_group(g_high, "高/低唤起（HighArousal=1/0）")
        dump_group(g_sent, "情绪倾向（Positive vs Negative）")
        dump_group(g_q,    "标题含问号（HasQ 1/0）")
        dump_group(g_excl, "标题含感叹号（HasExcl 1/0）")

        # 处理摘要
        f.write("## 处理与口径一致性摘要\n")
        f.write(f"- 比例变量越界裁剪样本数：{json.dumps(clip_summary, ensure_ascii=False)}\n")
        f.write(f"- 粉丝数 ≤ 0 的样本数（按 ln(1+fans) 口径处理）：{fans_nonpos}\n")
        f.write(f"- 未能映射唤起强度的情绪类别样本数（填充为 0，保持日志记录）：{unknown_arousal}\n")
        f.write(f"- 清洗后分析数据 CSV：{clean_csv}\n")
        f.write("\n备注：本报告仅作现象呈现与数据口径说明，不涉及因果推断；模型设定与估计见回归脚本。\n")

    # 10) 记录日志（供自查）
    log_path = os.path.join(paths["logs"], "P1_build_dataset_windows.log")
    info = {
        "input_path": input_path,
        "input_sha256": sha256_of_file(input_path),
        "rows": int(data.shape[0]),
        "clip_summary": clip_summary,
        "fans_nonpositive_count": int(fans_nonpos),
        "unknown_arousal_count_before_fill": int(unknown_arousal),
        "clean_csv_path": clean_csv,
        "outfile": outfile,
        "generated_at": now,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False, indent=2))

    print("[DONE] 单一文本报告已生成：", outfile)
    print("[DONE] 清洗后的分析数据 CSV：", clean_csv)
    print("[DONE] 日志：", log_path)
    return 0


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Windows 单文件：构建分析数据与统一 TXT 报告")
    ap.add_argument("--in", dest="input_path", type=str, default=DEFAULT_INPUT_PATH)
    ap.add_argument("--outfile", dest="outfile", type=str, default=DEFAULT_OUTFILE)
    ap.add_argument("--outdir", dest="outdir", type=str, default=DEFAULT_OUTDIR)
    args = ap.parse_args(argv)
    return run(args.input_path, args.outfile, args.outdir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
