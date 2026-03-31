"""
RandomForest 하이퍼파라미터 튜닝
- Fold 5 (2024~2025) 기준으로 빠르게 탐색
- 최적 파라미터로 Walk-Forward 재검증
- 최종 모델 저장
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

FOLDS = [
    ("2005-01-01", "2015-12-31", "2016-01-01", "2017-12-31"),
    ("2005-01-01", "2017-12-31", "2018-01-01", "2019-12-31"),
    ("2005-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
    ("2005-01-01", "2021-12-31", "2022-01-01", "2023-12-31"),
    ("2005-01-01", "2023-12-31", "2024-01-01", "2025-12-31"),
]

# 탐색할 파라미터 조합
PARAM_GRID = [
    {"n_estimators": 300,  "max_depth": 6,  "min_samples_leaf": 10, "max_features": "sqrt"},
    {"n_estimators": 300,  "max_depth": 8,  "min_samples_leaf": 10, "max_features": "sqrt"},
    {"n_estimators": 300,  "max_depth": 10, "min_samples_leaf": 10, "max_features": "sqrt"},
    {"n_estimators": 500,  "max_depth": 6,  "min_samples_leaf": 20, "max_features": "sqrt"},
    {"n_estimators": 500,  "max_depth": 8,  "min_samples_leaf": 20, "max_features": "sqrt"},
    {"n_estimators": 500,  "max_depth": 10, "min_samples_leaf": 20, "max_features": "sqrt"},
    {"n_estimators": 500,  "max_depth": 8,  "min_samples_leaf": 10, "max_features": "log2"},
    {"n_estimators": 500,  "max_depth": 8,  "min_samples_leaf": 30, "max_features": "sqrt"},
    {"n_estimators": 1000, "max_depth": 8,  "min_samples_leaf": 20, "max_features": "sqrt"},
    {"n_estimators": 1000, "max_depth": 10, "min_samples_leaf": 20, "max_features": "sqrt"},
]


def get_feat_cols(df):
    skip = {"ticker", "date", "label", "entry", "r5", "r10"}
    return [c for c in df.columns if c not in skip]


def eval_params(params, tr, te, feat_cols):
    model = RandomForestClassifier(
        **params, n_jobs=-1, random_state=42, class_weight="balanced"
    )
    model.fit(tr[feat_cols].values.astype(np.float32), tr["label"].values)
    pred = model.predict_proba(te[feat_cols].values.astype(np.float32))[:, 1]
    return roc_auc_score(te["label"].values, pred)


if __name__ == "__main__":
    df = pd.read_csv("lgbm_raw_us.csv", encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"로드: {len(df)}건 (양성:{(df.label==1).sum()} 음성:{(df.label==0).sum()})")

    FEAT_COLS = get_feat_cols(df)
    print(f"피처 수: {len(FEAT_COLS)}")

    # ── 1단계: Fold5만으로 빠른 탐색 ──────────────
    print(f"\n{'='*55}")
    print("[ 1단계: Fold5 기반 파라미터 탐색 ]")
    print(f"{'='*55}")

    tr_s, te_s = "2005-01-01", "2023-12-31"
    te_e_s     = "2025-12-31"
    tr5 = df[(df.date >= tr_s) & (df.date <= te_s)]
    te5 = df[(df.date >  te_s) & (df.date <= te_e_s)]
    print(f"train: {len(tr5)}건 / test: {len(te5)}건\n")

    search_results = []
    for i, params in enumerate(PARAM_GRID):
        auc = eval_params(params, tr5, te5, FEAT_COLS)
        search_results.append((auc, params))
        print(f"[{i+1:2d}/{len(PARAM_GRID)}] AUC:{auc:.4f} | {params}")

    search_results.sort(reverse=True)
    best_params = search_results[0][1]
    print(f"\n🏆 최적 파라미터: {best_params}")
    print(f"   Fold5 AUC: {search_results[0][0]:.4f}")
    print(f"\nTop 3:")
    for auc, p in search_results[:3]:
        print(f"  AUC:{auc:.4f} | {p}")

    # ── 2단계: 최적 파라미터로 전체 Walk-Forward ──
    print(f"\n{'='*55}")
    print("[ 2단계: 최적 파라미터 Walk-Forward 검증 ]")
    print(f"{'='*55}")

    fold_aucs   = []
    oos_records = []

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(FOLDS):
        tr = df[(df.date >= tr_start) & (df.date <= tr_end)]
        te = df[(df.date >= te_start) & (df.date <= te_end)]
        if len(tr) < 100 or len(te) < 50:
            print(f"Fold {i+1}: 스킵")
            continue

        model = RandomForestClassifier(
            **best_params, n_jobs=-1, random_state=42, class_weight="balanced"
        )
        model.fit(tr[FEAT_COLS].values.astype(np.float32), tr["label"].values)
        pred = model.predict_proba(te[FEAT_COLS].values.astype(np.float32))[:, 1]
        auc  = roc_auc_score(te["label"].values, pred)
        fold_aucs.append(auc)
        print(f"Fold {i+1}: {te_start[:4]}~{te_end[:4]} ({len(te)}건) AUC: {auc:.4f}")

        for j, (_, row) in enumerate(te.iterrows()):
            oos_records.append({
                "date":      row["date"],
                "ticker":    row["ticker"],
                "label":     int(row["label"]),
                "r5":        row.get("r5"),
                "r10":       row.get("r10"),
                "rf_score":  round(float(pred[j]), 4),
                "fold":      i + 1,
            })

    print(f"\n평균 AUC: {np.mean(fold_aucs):.4f} (±{np.std(fold_aucs):.4f})")
    print(f"Fold별: {[round(a,4) for a in fold_aucs]}")

    # ── OOS 분석 ──────────────────────────────────
    if oos_records:
        oos_df = pd.DataFrame(oos_records)
        bins   = [0, 0.1, 0.15, 0.2, 0.3, 0.4, 1.0]
        labels = ["~0.1","0.1~0.15","0.15~0.2","0.2~0.3","0.3~0.4","0.4+"]
        oos_df["score_bin"] = pd.cut(oos_df.rf_score, bins=bins, labels=labels)

        print(f"\n[ OOS 점수 분석 ] 총 {len(oos_df)}건")
        print(f"점수 분포: 평균={oos_df.rf_score.mean():.3f} "
              f"max={oos_df.rf_score.max():.3f}")
        print(f"{'구간':12s} {'n':>5} {'승률':>8} {'평균r5':>8} {'평균r10':>8}")
        print("-" * 50)
        for lbl in labels:
            sub = oos_df[oos_df.score_bin == lbl]
            if len(sub) == 0: continue
            r10 = sub["r10"].dropna()
            r5  = sub["r5"].dropna()
            win = (r10 >= 8).mean() * 100 if len(r10) > 0 else 0
            print(f"{lbl:12s} {len(sub):>5} {win:>7.1f}% "
                  f"{r5.mean():>+7.1f}% {r10.mean():>+7.1f}%")
        r10_all = oos_df["r10"].dropna()
        print("-" * 50)
        print(f"{'전체':12s} {len(oos_df):>5} "
              f"{(r10_all>=8).mean()*100:>7.1f}% "
              f"{oos_df['r5'].dropna().mean():>+7.1f}% "
              f"{r10_all.mean():>+7.1f}%")

        oos_df.to_csv("rf_oos_us.csv", index=False, encoding="utf-8-sig")

    # ── 최종 모델 저장 ─────────────────────────────
    print(f"\n최종 모델 학습 중...")
    tr_f = df[df.date < "2025-01-01"]
    te_f = df[df.date >= "2025-01-01"]
    if len(te_f) < 50:
        n = int(len(df) * 0.1)
        tr_f = df.iloc[:-n]; te_f = df.iloc[-n:]

    print(f"train: {len(tr_f)}건 / val: {len(te_f)}건")
    final = RandomForestClassifier(
        **best_params, n_jobs=-1, random_state=42, class_weight="balanced"
    )
    final.fit(tr_f[FEAT_COLS].values.astype(np.float32), tr_f["label"].values)
    val_auc = roc_auc_score(
        te_f["label"].values,
        final.predict_proba(te_f[FEAT_COLS].values.astype(np.float32))[:, 1]
    )
    print(f"최종 Val AUC: {val_auc:.4f}")

    with open("model_rf_us.pkl", "wb") as f:
        pickle.dump(final, f)
    with open("feat_cols_lgbm_us.pkl", "wb") as f:
        pickle.dump(FEAT_COLS, f)
    with open("best_params_rf.txt", "w") as f:
        f.write(str(best_params))

    print(f"저장 완료: model_rf_us.pkl")
    print(f"Walk-Forward 평균 AUC: {np.mean(fold_aucs):.4f}")
