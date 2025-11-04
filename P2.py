#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.py — Windows 单文件版：回归估计 + 诊断 + 稳健性 + 非线性（统一 TXT 报告）
用途：论文附录·文件二（整合版；无图）
- 读取 P1.py 生成的 analytic_dataset.csv
- 基准 OLS：四个因变量（点赞率/投币率/收藏率/弹幕密度），统一解释变量与口径
- 畸变稳健：异方差稳健标准误（HC1）
- 诊断：Breusch–Pagan、White 异方差检验；Cook’s D/杠杆影响点 Top 10
- 稳健性：对因变量 1%–99% Winsorize 后复估并并列表达核心系数
- 非线性：标题长度二次项“倒U”拐点及 Delta 方法 95%CI
- VIF：核心数值变量（不含分区虚拟变量）
依赖：
  pip install pandas statsmodels numpy
"""

from __future__ import annotations
import argparse, hashlib, json, os, sys, datetime, math, warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels.stats.outliers_influence")

# ===================== 默认 Windows 路径（可用命令行覆盖） =====================
DEFAULT_DATA_CSV = r"C:\Users\86173\Desktop\data\clean\analytic_dataset.csv"
DEFAULT_OUTFILE  = r"C:\Users\86173\Desktop\Appendix_Model_Report.txt"

# ===================== 工具函数 =====================
def sha256_of_file(path: str) -> str:
    if not os.path.exists(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()

def fmt(x, nd=4) -> str:
    try:
        if x is None: return "NA"
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return "inf" if np.isinf(xf) else "NA"
        s = f"{xf:.{nd}f}"
        if s.startswith("-0.0000"): s = s.replace("-0.0000", "0.0000")
        if s.startswith("-0.000"): s = s.replace("-0.000", "0.000")
        return s
    except Exception:
        return "NA"

def stars(p):
    try:
        p = float(p)
    except:
        return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

def winsorize_series(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    ql = x.quantile(lo)
    qh = x.quantile(hi)
    return x.clip(ql, qh)

def add_constant(X: pd.DataFrame) -> pd.DataFrame:
    if "const" not in X.columns:
        X = sm.add_constant(X, has_constant="add")
    return X

def mode_value(s: pd.Series) -> str:
    s = s.dropna().astype(str)
    if s.empty: return ""
    return s.mode(dropna=True).iloc[0]

# ===================== 建模核心 =====================
def build_design(df: pd.DataFrame, y_col: str, ref_cat: str) -> Tuple[pd.Series, pd.DataFrame, Dict]:
    """
    构建设计矩阵：
      Y = β0 + β1 HighArousal + β2 Positive + β3 Negative
          + β4 TitleLen + β5 TitleLen^2 + β6 HasQ + β7 HasExcl
          + γ1 ln_fans + γ2 duration_y + δ_category + ε
      分区 category_y 做 Treatment 编码，参照组为 ref_cat（众数）
    """
    # 数值列
    for c in ["TitleLen", "ln_fans", "duration_y"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 二元列
    for c in ["HighArousal", "Positive", "Negative", "HasQ", "HasExcl"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # 二次项
    df["TitleLen2"] = (df["TitleLen"] ** 2).astype(float)

    # 分区——设置参照组为众数
    cats = [ref_cat] + [c for c in df["category_y"].astype(str).unique().tolist() if c != ref_cat]
    df["category_y"] = pd.Categorical(df["category_y"].astype(str), categories=cats)
    dummies = pd.get_dummies(df["category_y"], prefix="CAT", drop_first=True)

    # 组合 X
    X_core_cols = ["HighArousal", "Positive", "Negative", "TitleLen", "TitleLen2", "HasQ", "HasExcl", "ln_fans", "duration_y"]
    X = pd.concat([df[X_core_cols], dummies], axis=1)
    X = add_constant(X)

    # 目标变量
    y = pd.to_numeric(df[y_col], errors="coerce")

    # 丢弃缺失
    data = pd.concat([y, X], axis=1).dropna()
    y = data[y_col].astype(float)
    X = data.drop(columns=[y_col]).astype(float)

    meta = {
        "ref_category": ref_cat,
        "n_cats": int(dummies.shape[1] + 1),
        "y_col": y_col,
        "n_obs": int(y.shape[0]),
        "titlelen_range": (float(df["TitleLen"].min()), float(df["TitleLen"].max()))
    }
    return y, X, meta

def fit_ols_hc1(y: pd.Series, X: pd.DataFrame):
    """
    拟合 OLS，并返回带 HC1 稳健协方差的结果对象（robust_res）
    """
    model = sm.OLS(y, X)
    res = model.fit()
    robust_res = res.get_robustcov_results(cov_type="HC1")
    return res, robust_res

def table_from_result(robust_res, pretty_map: Dict[str, str]) -> List[Dict]:
    """
    从稳健结果构造系数表（兼容 params 为 ndarray）
    """
    names = list(robust_res.model.exog_names)
    coefs = np.asarray(robust_res.params)
    ses   = np.asarray(robust_res.bse)
    ts    = np.asarray(robust_res.tvalues)
    ps    = np.asarray(robust_res.pvalues)

    rows = []
    for i, name in enumerate(names):
        label = "常数项" if name == "const" else pretty_map.get(name, name)
        rows.append({
            "var": label,
            "coef": coefs[i],
            "se": ses[i],
            "t": ts[i],
            "p": ps[i],
            "stars": stars(ps[i]),
        })
    return rows

def delta_turning_point(robust_res, b1_name="TitleLen", b2_name="TitleLen2") -> Dict:
    """
    倒U拐点 g = -β1/(2β2)，Delta 方法近似方差。
    兼容 cov_params() 返回 ndarray 的情况。
    """
    names = list(robust_res.model.exog_names)
    if b1_name not in names or b2_name not in names:
        return {"ok": False, "msg": "缺少 TitleLen/TitleLen2 系数，无法计算拐点"}

    i1 = names.index(b1_name)
    i2 = names.index(b2_name)
    params = np.asarray(robust_res.params)
    cov    = np.asarray(robust_res.cov_params())

    b1 = params[i1]
    b2 = params[i2]
    if b2 == 0 or np.isnan(b1) or np.isnan(b2):
        return {"ok": False, "msg": "TitleLen2 系数为 0 或 NaN，无法计算拐点"}

    g = -b1 / (2.0 * b2)
    # 梯度
    dg_db1 = -1.0 / (2.0 * b2)
    dg_db2 = b1 / (2.0 * (b2 ** 2))
    # 协方差元素
    v11 = cov[i1, i1]
    v22 = cov[i2, i2]
    v12 = cov[i1, i2]
    var_g = (dg_db1 ** 2) * v11 + (dg_db2 ** 2) * v22 + 2.0 * dg_db1 * dg_db2 * v12
    se_g = float(np.sqrt(var_g)) if var_g >= 0 else np.nan
    lo = g - 1.96 * se_g if not np.isnan(se_g) else np.nan
    hi = g + 1.96 * se_g if not np.isnan(se_g) else np.nan
    return {"ok": True, "turning": float(g), "se": float(se_g), "lo95": float(lo), "hi95": float(hi)}

def run_vif_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    对核心数值变量（不含分类虚拟变量）计算 VIF：
    HighArousal, Positive, Negative, TitleLen, TitleLen2, HasQ, HasExcl, ln_fans, duration_y
    """
    cols = ["HighArousal", "Positive", "Negative", "TitleLen", "TitleLen2", "HasQ", "HasExcl", "ln_fans", "duration_y"]
    X = df[cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.dropna()
    X = add_constant(X)

    vif_rows = []
    # 静默掉内部的 divide by zero warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, c in enumerate(X.columns):
            if c == "const":
                continue
            try:
                vif_val = variance_inflation_factor(X.values, i)
            except Exception:
                vif_val = np.nan
            vif_rows.append({"variable": c, "VIF": vif_val})
    return pd.DataFrame(vif_rows)

def dump_rows_as_table(f, rows: List[Dict], headers: List[Tuple[str, int]]):
    totalw = sum(w for _, w in headers) + (len(headers) - 1) * 3
    f.write("-" * totalw + "\n")
    f.write(" | ".join([f"{h:<{w}}" for h, w in headers]) + "\n")
    f.write("-" * totalw + "\n")
    for r in rows:
        f.write(" | ".join([f"{str(r.get(h,'')):<{w}}" for h, w in headers]) + "\n")
    f.write("-" * totalw + "\n\n")

# ===================== 主程序 =====================
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Windows 单文件：回归估计 + 诊断 + 稳健性（统一 TXT 报告）")
    ap.add_argument("--data", dest="data_csv", type=str, default=DEFAULT_DATA_CSV)
    ap.add_argument("--outfile", dest="outfile", type=str, default=DEFAULT_OUTFILE)
    args = ap.parse_args(argv)

    if not os.path.exists(args.data_csv):
        print(f"[ERROR] 找不到数据文件：{args.data_csv}")
        return 1

    # 读取数据
    df = pd.read_csv(args.data_csv, encoding="utf-8-sig")
    df["TitleLen"] = pd.to_numeric(df["TitleLen"], errors="coerce")
    df["ln_fans"] = pd.to_numeric(df["ln_fans"], errors="coerce")
    df["duration_y"] = pd.to_numeric(df["duration_y"], errors="coerce")
    for c in ["HighArousal", "Positive", "Negative", "HasQ", "HasExcl"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    # 保证四个因变量为数值
    y_list = ["ratio_like_y", "ratio_coin_y", "ratio_fav_y", "ratio_danmaku_y"]
    for y in y_list:
        df[y] = pd.to_numeric(df[y], errors="coerce")

    # 参照分区（众数）
    ref_cat = mode_value(df["category_y"])
    # 预先构造 TitleLen2 供 VIF
    df["TitleLen2"] = df["TitleLen"] ** 2

    # 输出
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(args.outfile, "w", encoding="utf-8") as f:
        f.write("# 附录：模型设定、估计与诊断（统一文本报告）\n")
        f.write(f"生成时间：{now}\n\n")
        f.write("## 数据与设定\n")
        f.write(f"- 输入数据：{args.data_csv}\n")
        f.write(f"- 数据 SHA256：{sha256_of_file(args.data_csv)}\n")
        f.write(f"- 总样本行数（含缺失）：{df.shape[0]}\n")
        f.write(f"- 分区参照组（众数）：{ref_cat}\n\n")

        f.write("## 模型设定（四个因变量共用）\n")
        f.write("Y = β0 + β1 HighArousal + β2 Positive + β3 Negative + β4 TitleLen + β5 TitleLen^2 + β6 HasQ + β7 HasExcl + γ1 ln(fans) + γ2 duration + δ_category + ε\n")
        f.write("估计方法：OLS + 异方差稳健标准误（HC1）。分类变量 `category_y` 采用 Treatment 编码，参照组为众数；中性情绪为参照（Positive/Negative 为双侧二分）。\n\n")

        # VIF（仅数值变量）
        f.write("## 多重共线性（VIF）—核心数值变量（不含分区虚拟变量）\n")
        vif_df = run_vif_numeric(df)
        # 美化 VIF 数字
        vif_rows = []
        for _, r in vif_df.iterrows():
            vif_rows.append({"variable": r["variable"], "VIF": fmt(r["VIF"])})
        dump_rows_as_table(f, vif_rows, [("variable", 18), ("VIF", 12)])
        f.write("注：分类虚拟变量不纳入 VIF 以避免病态共线；VIF>10 可视为较强共线迹象（经验规则）。\n\n")

        # 逐个因变量建模
        for y_col in y_list:
            f.write(f"## 模型结果 — 因变量：{y_col}\n")
            y, X, meta = build_design(df.copy(), y_col=y_col, ref_cat=ref_cat)

            # 拟合
            res, rob = fit_ols_hc1(y, X)

            # 美化变量名
            pretty = {
                "HighArousal": "高唤起",
                "Positive": "正向情绪",
                "Negative": "负向情绪",
                "TitleLen": "标题长度",
                "TitleLen2": "标题长度^2",
                "HasQ": "标题含问号",
                "HasExcl": "标题含感叹号",
                "ln_fans": "ln(粉丝数)",
                "duration_y": "时长(秒)",
            }
            for c in X.columns:
                if c.startswith("CAT_"):
                    pretty[c] = f"内容分区={c.replace('CAT_', '')}"

            # 系数表
            rows = table_from_result(rob, pretty)
            # 转字符串
            for r in rows:
                r["coef"], r["se"], r["t"], r["p"] = fmt(r["coef"]), fmt(r["se"]), fmt(r["t"]), fmt(r["p"])
            dump_rows_as_table(
                f, rows, [("var", 22), ("coef", 12), ("se", 12), ("t", 10), ("p", 10), ("stars", 6)]
            )

            # 拟合优度
            f.write(f"R^2：{fmt(res.rsquared)}    调整后 R^2：{fmt(res.rsquared_adj)}    样本量 N：{int(rob.nobs)}\n\n")

            # 诊断：BP/White
            try:
                bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(res.resid, X)
                f.write(f"Breusch–Pagan: LM={fmt(bp_stat,3)}, p={fmt(bp_p,4)}; F={fmt(bp_f,3)}, p={fmt(bp_fp,4)}\n")
            except Exception as e:
                f.write(f"Breusch–Pagan: 计算失败（{e}）\n")
            try:
                w_stat, w_p, f_stat, f_p = het_white(res.resid, X)
                f.write(f"White: LM={fmt(w_stat,3)}, p={fmt(w_p,4)}; F={fmt(f_stat,3)}, p={fmt(f_p,4)}\n\n")
            except Exception as e:
                f.write(f"White: 计算失败（{e}）\n\n")

            # 影响点（使用 get_influence 以避免兼容性问题）
            try:
                infl = res.get_influence()
                cooks_d = infl.cooks_distance[0]
                leverage = infl.hat_matrix_diag
                idx = np.argsort(-cooks_d)[:10]
                infl_rows = []
                for i in idx:
                    infl_rows.append({"idx": int(i), "CookD": fmt(cooks_d[i]), "Leverage": fmt(leverage[i])})
                f.write("影响点（Cook's D Top 10）\n")
                dump_rows_as_table(f, infl_rows, [("idx", 8), ("CookD", 12), ("Leverage", 12)])
            except Exception as e:
                f.write(f"影响点分析失败：{e}\n\n")

            # 非线性：倒U拐点
            tp = delta_turning_point(rob)
            if tp.get("ok", False):
                lo, hi = tp["lo95"], tp["hi95"]
                rng = meta["titlelen_range"]
                within = (tp["turning"] >= rng[0]) and (tp["turning"] <= rng[1])
                f.write("标题长度‘倒U’拐点（基于 TitleLen & TitleLen^2）：\n")
                f.write(f"- 拐点位置：{fmt(tp['turning'],3)}（95%CI: {fmt(lo,3)} ~ {fmt(hi,3)}），"
                        f"是否落在样本范围[{fmt(rng[0],0)}, {fmt(rng[1],0)}]：{'是' if within else '否'}\n\n")
            else:
                f.write(f"标题长度‘倒U’拐点：{tp.get('msg','无法计算')}\n\n")

            # 稳健性：Winsorize 因变量 1%–99%
            y_w = winsorize_series(y, 0.01, 0.99)
            res_w = sm.OLS(y_w, X).fit().get_robustcov_results(cov_type="HC1")
            key_vars = ["HighArousal", "Positive", "Negative", "TitleLen", "TitleLen2"]
            rob_rows = []
            names = list(rob.model.exog_names)
            for vn in key_vars:
                if vn in names:
                    i = names.index(vn)
                    rob_rows.append({
                        "var": pretty.get(vn, vn),
                        "coef_base": fmt(rob.params[i]),
                        "p_base": fmt(rob.pvalues[i], 4),
                        "coef_winsor": fmt(res_w.params[i]),
                        "p_winsor": fmt(res_w.pvalues[i], 4),
                    })
            f.write("稳健性对比（因变量 Winsorize 1%–99% 后）\n")
            dump_rows_as_table(f, rob_rows, [("var", 22), ("coef_base", 14), ("p_base", 10), ("coef_winsor", 14), ("p_winsor", 10)])

        # 复现信息
        f.write("## 复现实验信息\n")
        import platform
        f.write(f"- Python：{platform.python_version()}  pandas：{pd.__version__}  statsmodels：{sm.__version__}\n")
        f.write("- 协方差口径：HC1；稳健性处理仅改造因变量（Winsorize 1%–99%），解释变量口径不变；分类参照组为 `category_y` 众数。\n")

    print("[DONE] 模型统一文本报告已生成：", args.outfile)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
