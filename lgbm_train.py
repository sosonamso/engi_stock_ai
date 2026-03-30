"""
미장 LightGBM 학습 (EODHD 버전)
- lgbm_raw_us.csv 로드
- Walk-Forward 검증
- OOS 점수 분석
- 최종 모델 저장
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

FOLDS = [
    ("2005-01-01", "2015-12-31", "2016-01-01", "2017-12-31"),
    ("2005-01-01", "2017-12-31", "2018-01-01", "2019-12-31"),
    ("2005-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
    ("2005-01-01", "2021-12-31", "2022-01-01", "2023-12-31"),
    ("2005-01-01", "2023-12-31", "2024-01-01", "2025-12-31"),
]

LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "auc",
    "boosting_type":     "gbdt",
    "num_leaves":        63,
    "learning_rate":     0.05,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "min_child_samples": 20,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "verbose":           -1,
    "n_jobs":            -1,
}


def get_feat_cols(df):
    skip = {"ticker", "date", "label", "entry", "r5", "r10"}
    return [c for c in df.columns if c not in skip]


if __name__ == "__main__":
    df = pd.read_csv("lgbm_raw_us.csv", encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"로드: {len(df)}건 (양성:{(df.label==1).sum()} 음성:{(df.label==0).sum()})")
    print(f"날짜: {df.date.min().date()} ~ {df.date.max().date()}")

    FEAT_COLS = get_feat_cols(df)
    print(f"피처 수: {len(FEAT_COLS)}")

    # ── Walk-Forward Validation ────────────────────────
    fold_aucs    = []
    feat_imp_list = []
    oos_records  = []

    print(f"\n{'='*55}")
    print("[ 미장 LightGBM Walk-Forward ]")
    print(f"{'='*55}")

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(FOLDS):
        tr = df[(df.date >= tr_start) & (df.date <= tr_end)]
        te = df[(df.date >= te_start) & (df.date <= te_end)]

        if len(tr) < 100 or len(te) < 50:
            print(f"Fold {i+1}: 데이터 부족 스킵 (train:{len(tr)} test:{len(te)})")
            continue

        X_tr = tr[FEAT_COLS].values.astype(np.float32)
        y_tr = tr["label"].values
        X_te = te[FEAT_COLS].values.astype(np.float32)
        y_te = te["label"].values

        print(f"\nFold {i+1}: {tr_start[:4]}~{tr_end[:4]} ({len(tr)}건) → test {te_start[:4]}~{te_end[:4]} ({len(te)}건)")
        print(f"  양성비율 train:{y_tr.mean():.2f} test:{y_te.mean():.2f}")

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEAT_COLS)
        dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)

        model = lgb.train(
            LGB_PARAMS, dtrain, num_boost_round=1000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )

        pred = model.predict(X_te)
        auc  = roc_auc_score(y_te, pred)
        fold_aucs.append(auc)
        print(f"  → AUC: {auc:.4f} (best iter: {model.best_iteration})")

        imp = pd.Series(model.feature_importance(importance_type="gain"), index=FEAT_COLS)
        feat_imp_list.append(imp)

        # OOS 기록
        for j, (_, row) in enumerate(te.iterrows()):
            oos_records.append({
                "date":       row["date"],
                "ticker":     row["ticker"],
                "label":      int(row["label"]),
                "r5":         row.get("r5"),
                "r10":        row.get("r10"),
                "lgbm_score": round(float(pred[j]), 4),
                "fold":       i + 1,
            })

    print(f"\n평균 AUC: {np.mean(fold_aucs):.4f} (±{np.std(fold_aucs):.4f})")
    print(f"Fold별: {[round(a,4) for a in fold_aucs]}")

    # 피처 중요도
    if feat_imp_list:
        avg_imp = pd.concat(feat_imp_list, axis=1).mean(axis=1).sort_values(ascending=False)
        print(f"\n피처 중요도 Top 20:")
        for feat, val in avg_imp.head(20).items():
            print(f"  {feat:30s} {val:.1f}")

    # ── OOS 점수 분석 ──────────────────────────────────
    if oos_records:
        oos_df = pd.DataFrame(oos_records)
        print(f"\n{'='*55}")
        print(f"[ OOS 점수 분석 ] 총 {len(oos_df)}건")
        print(f"점수 분포: 평균={oos_df.lgbm_score.mean():.3f} "
              f"min={oos_df.lgbm_score.min():.3f} max={oos_df.lgbm_score.max():.3f}")

        bins   = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
        labels = ["~0.2","0.2~0.3","0.3~0.4","0.4~0.5","0.5~0.6","0.6+"]
        oos_df["score_bin"] = pd.cut(oos_df.lgbm_score, bins=bins, labels=labels)

        print(f"\n{'구간':10s} {'n':>5} {'승률':>8} {'평균r5':>8} {'평균r10':>8}")
        print("-" * 45)
        for lbl in labels:
            sub  = oos_df[oos_df.score_bin == lbl]
            if len(sub) == 0: continue
            r10  = sub["r10"].dropna()
            r5   = sub["r5"].dropna()
            win  = (r10 >= 8).mean() * 100 if len(r10) > 0 else 0
            print(f"{lbl:10s} {len(sub):>5} {win:>7.1f}% "
                  f"{r5.mean():>+7.1f}% {r10.mean():>+7.1f}%")

        r10_all = oos_df["r10"].dropna()
        r5_all  = oos_df["r5"].dropna()
        print("-" * 45)
        print(f"{'전체':10s} {len(oos_df):>5} "
              f"{(r10_all>=8).mean()*100:>7.1f}% "
              f"{r5_all.mean():>+7.1f}% {r10_all.mean():>+7.1f}%")

        oos_df.to_csv("lgbm_oos_us.csv", index=False, encoding="utf-8-sig")
        print("\nOOS 결과 저장: lgbm_oos_us.csv")

    # ── 최종 모델 (시간 기준: ~2024 train / 2025 val) ──
    print("\n최종 모델 학습 중 (train: ~2024 / val: 2025~)...")
    tr_f = df[df.date <  "2025-01-01"]
    te_f = df[df.date >= "2025-01-01"]

    if len(te_f) < 50:
        n_val = int(len(df) * 0.1)
        tr_f  = df.iloc[:-n_val]
        te_f  = df.iloc[-n_val:]

    print(f"  train: {len(tr_f)}건 / val: {len(te_f)}건")

    dtrain_f = lgb.Dataset(
        tr_f[FEAT_COLS].values.astype(np.float32),
        label=tr_f["label"].values, feature_name=FEAT_COLS)
    dvalid_f = lgb.Dataset(
        te_f[FEAT_COLS].values.astype(np.float32),
        label=te_f["label"].values, reference=dtrain_f)

    final_model = lgb.train(
        LGB_PARAMS, dtrain_f, num_boost_round=1000,
        valid_sets=[dvalid_f],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
    )

    val_auc = roc_auc_score(
        te_f["label"].values,
        final_model.predict(te_f[FEAT_COLS].values.astype(np.float32))
    )
    print(f"최종 모델 Val AUC: {val_auc:.4f}")

    final_model.save_model("model_lgbm_us.txt")
    with open("feat_cols_lgbm_us.pkl", "wb") as f:
        pickle.dump(FEAT_COLS, f)
    print("저장: model_lgbm_us.txt / feat_cols_lgbm_us.pkl")
    print(f"\nWalk-Forward 평균 AUC: {np.mean(fold_aucs):.4f}")
