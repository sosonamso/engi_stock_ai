"""
멀티 모델 검증
- LightGBM / XGBoost / RandomForest / LSTM
- 동일 데이터로 Walk-Forward AUC 비교
- 최종 모델 저장
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FOLDS = [
    ("2005-01-01", "2015-12-31", "2016-01-01", "2017-12-31"),
    ("2005-01-01", "2017-12-31", "2018-01-01", "2019-12-31"),
    ("2005-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
    ("2005-01-01", "2021-12-31", "2022-01-01", "2023-12-31"),
    ("2005-01-01", "2023-12-31", "2024-01-01", "2025-12-31"),
]

LGB_PARAMS = {
    "objective": "binary", "metric": "auc",
    "num_leaves": 63, "learning_rate": 0.05,
    "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 5, "min_child_samples": 20,
    "reg_alpha": 0.1, "reg_lambda": 0.1,
    "verbose": -1, "n_jobs": -1,
}

XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 20, "reg_alpha": 0.1, "reg_lambda": 0.1,
    "n_jobs": -1, "verbosity": 0,
}

LSTM_SEQ    = 150   # 시계열 길이
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
LSTM_EPOCHS = 20
LSTM_BATCH  = 256


def get_feat_cols(df):
    skip = {"ticker", "date", "label", "entry", "r5", "r10"}
    return [c for c in df.columns if c not in skip]


def get_seq_cols(feat_cols):
    """LSTM용 시계열 피처 (ret, close_norm, vol_ratio × 150일)"""
    seq = []
    for k in range(1, LSTM_SEQ + 1):
        for prefix in ["ret_", "close_norm_", "vol_ratio_"]:
            col = f"{prefix}{k}"
            if col in feat_cols:
                seq.append(col)
    return seq


def get_scalar_cols(feat_cols, seq_cols):
    """LSTM용 스칼라 피처"""
    return [c for c in feat_cols if c not in seq_cols]


# ── LightGBM ──────────────────────────────────────
def train_lgbm(X_tr, y_tr, X_te, y_te, feat_cols):
    import lightgbm as lgb
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols)
    dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
    model  = lgb.train(
        LGB_PARAMS, dtrain, num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    )
    pred = model.predict(X_te)
    return roc_auc_score(y_te, pred), model, pred


# ── XGBoost ───────────────────────────────────────
def train_xgb(X_tr, y_tr, X_te, y_te):
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_te, label=y_te)
    model  = xgb.train(
        XGB_PARAMS, dtrain, num_boost_round=1000,
        evals=[(dvalid, "val")],
        early_stopping_rounds=50, verbose_eval=False
    )
    pred = model.predict(xgb.DMatrix(X_te))
    return roc_auc_score(y_te, pred), model, pred


# ── Random Forest ─────────────────────────────────
def train_rf(X_tr, y_tr, X_te, y_te):
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8,
        min_samples_leaf=20, n_jobs=-1, random_state=42
    )
    model.fit(X_tr, y_tr)
    pred = model.predict_proba(X_te)[:, 1]
    return roc_auc_score(y_te, pred), model, pred


# ── LSTM ──────────────────────────────────────────
def train_lstm(X_seq_tr, X_sc_tr, y_tr, X_seq_te, X_sc_te, y_te):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        class StockLSTM(nn.Module):
            def __init__(self, seq_feat, sc_feat, hidden, layers):
                super().__init__()
                self.lstm = nn.LSTM(seq_feat, hidden, layers,
                                    batch_first=True, dropout=0.3)
                self.fc   = nn.Sequential(
                    nn.Linear(hidden + sc_feat, 64),
                    nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(64, 1), nn.Sigmoid()
                )
            def forward(self, seq, sc):
                _, (h, _) = self.lstm(seq)
                h = h[-1]
                x = torch.cat([h, sc], dim=1)
                return self.fc(x).squeeze(1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        seq_feat = X_seq_tr.shape[2]
        sc_feat  = X_sc_tr.shape[1]

        model = StockLSTM(seq_feat, sc_feat, LSTM_HIDDEN, LSTM_LAYERS).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()

        ds = TensorDataset(
            torch.FloatTensor(X_seq_tr),
            torch.FloatTensor(X_sc_tr),
            torch.FloatTensor(y_tr)
        )
        loader = DataLoader(ds, batch_size=LSTM_BATCH, shuffle=True)

        model.train()
        for epoch in range(LSTM_EPOCHS):
            for seq_b, sc_b, y_b in loader:
                seq_b, sc_b, y_b = seq_b.to(device), sc_b.to(device), y_b.to(device)
                pred = model(seq_b, sc_b)
                loss = loss_fn(pred, y_b)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(
                torch.FloatTensor(X_seq_te).to(device),
                torch.FloatTensor(X_sc_te).to(device)
            ).cpu().numpy()

        return roc_auc_score(y_te, pred), model, pred

    except Exception as e:
        print(f"  LSTM 실패: {e}")
        return None, None, None


if __name__ == "__main__":
    df = pd.read_csv("lgbm_raw_us.csv", encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"로드: {len(df)}건 (양성:{(df.label==1).sum()} 음성:{(df.label==0).sum()})")
    print(f"날짜: {df.date.min().date()} ~ {df.date.max().date()}")

    FEAT_COLS  = get_feat_cols(df)
    SEQ_COLS   = get_seq_cols(FEAT_COLS)
    SCALAR_COLS = get_scalar_cols(FEAT_COLS, SEQ_COLS)
    print(f"전체 피처: {len(FEAT_COLS)} | 시계열: {len(SEQ_COLS)} | 스칼라: {len(SCALAR_COLS)}")

    # 스칼라 피처 스케일러
    scaler = StandardScaler()

    results = {
        "LightGBM":     [],
        "XGBoost":      [],
        "RandomForest": [],
        "LSTM":         [],
    }
    best_models = {}
    oos_records = []

    print(f"\n{'='*60}")
    print("[ 멀티 모델 Walk-Forward Validation ]")
    print(f"{'='*60}")

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(FOLDS):
        tr = df[(df.date >= tr_start) & (df.date <= tr_end)]
        te = df[(df.date >= te_start) & (df.date <= te_end)]
        if len(tr) < 100 or len(te) < 50:
            print(f"Fold {i+1}: 스킵")
            continue

        print(f"\nFold {i+1}: {tr_start[:4]}~{tr_end[:4]} ({len(tr)}건) → {te_start[:4]}~{te_end[:4]} ({len(te)}건)")

        X_tr = tr[FEAT_COLS].values.astype(np.float32)
        y_tr = tr["label"].values.astype(np.float32)
        X_te = te[FEAT_COLS].values.astype(np.float32)
        y_te = te["label"].values.astype(np.float32)

        # 스칼라 스케일링 (RF/LSTM용)
        X_sc_tr = scaler.fit_transform(tr[SCALAR_COLS].values.astype(np.float32))
        X_sc_te = scaler.transform(te[SCALAR_COLS].values.astype(np.float32))

        # LSTM 시계열 reshape: (N, 150, 3)
        n_steps = LSTM_SEQ
        n_feats = 3  # ret, close_norm, vol_ratio
        try:
            ret_cols   = [f"ret_{k}"        for k in range(1, n_steps+1)]
            norm_cols  = [f"close_norm_{k}" for k in range(1, n_steps+1)]
            vol_cols   = [f"vol_ratio_{k}"  for k in range(1, n_steps+1)]
            X_seq_tr = np.stack([
                tr[ret_cols].values,
                tr[norm_cols].values,
                tr[vol_cols].values
            ], axis=2).astype(np.float32)
            X_seq_te = np.stack([
                te[ret_cols].values,
                te[norm_cols].values,
                te[vol_cols].values
            ], axis=2).astype(np.float32)
            lstm_ok = True
        except:
            lstm_ok = False

        # LightGBM
        try:
            auc, model, pred = train_lgbm(X_tr, y_tr, X_te, y_te, FEAT_COLS)
            results["LightGBM"].append(auc)
            best_models["LightGBM"] = model
            print(f"  LightGBM  AUC: {auc:.4f}")
            for j, (_, row) in enumerate(te.iterrows()):
                oos_records.append({
                    "date": row["date"], "ticker": row["ticker"],
                    "label": int(row["label"]),
                    "r5": row.get("r5"), "r10": row.get("r10"),
                    "lgbm_score": round(float(pred[j]), 4), "fold": i+1
                })
        except Exception as e:
            print(f"  LightGBM  실패: {e}")

        # XGBoost
        try:
            auc, model, pred = train_xgb(X_tr, y_tr, X_te, y_te)
            results["XGBoost"].append(auc)
            best_models["XGBoost"] = model
            print(f"  XGBoost   AUC: {auc:.4f}")
        except Exception as e:
            print(f"  XGBoost   실패: {e}")

        # Random Forest
        try:
            auc, model, pred = train_rf(X_tr, y_tr, X_te, y_te)
            results["RandomForest"].append(auc)
            best_models["RandomForest"] = model
            print(f"  RF        AUC: {auc:.4f}")
        except Exception as e:
            print(f"  RF        실패: {e}")

        # LSTM
        if lstm_ok:
            auc, model, pred = train_lstm(X_seq_tr, X_sc_tr, y_tr,
                                          X_seq_te, X_sc_te, y_te)
            if auc is not None:
                results["LSTM"].append(auc)
                best_models["LSTM"] = model
                print(f"  LSTM      AUC: {auc:.4f}")

    # ── 결과 요약 ─────────────────────────────────
    print(f"\n{'='*60}")
    print("[ 모델별 Walk-Forward 평균 AUC ]")
    print(f"{'='*60}")
    best_model_name = None
    best_auc = 0
    for name, aucs in results.items():
        if not aucs: continue
        mean_auc = np.mean(aucs)
        std_auc  = np.std(aucs)
        print(f"  {name:15s} {mean_auc:.4f} (±{std_auc:.4f}) folds:{len(aucs)}")
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_model_name = name

    print(f"\n🏆 최고 성능: {best_model_name} (AUC {best_auc:.4f})")

    # ── OOS 분석 ──────────────────────────────────
    if oos_records:
        oos_df = pd.DataFrame(oos_records)
        bins   = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
        labels = ["~0.2","0.2~0.3","0.3~0.4","0.4~0.5","0.5~0.6","0.6+"]
        oos_df["score_bin"] = pd.cut(oos_df.lgbm_score, bins=bins, labels=labels)

        print(f"\n[ LightGBM OOS 점수 분석 ] 총 {len(oos_df)}건")
        print(f"{'구간':10s} {'n':>5} {'승률':>8} {'평균r5':>8} {'평균r10':>8}")
        print("-" * 45)
        for lbl in labels:
            sub = oos_df[oos_df.score_bin == lbl]
            if len(sub) == 0: continue
            r10 = sub["r10"].dropna()
            r5  = sub["r5"].dropna()
            win = (r10 >= 8).mean() * 100 if len(r10) > 0 else 0
            print(f"{lbl:10s} {len(sub):>5} {win:>7.1f}% "
                  f"{r5.mean():>+7.1f}% {r10.mean():>+7.1f}%")
        r10_all = oos_df["r10"].dropna()
        print("-" * 45)
        print(f"{'전체':10s} {len(oos_df):>5} "
              f"{(r10_all>=8).mean()*100:>7.1f}%")

    # ── 최종 모델 저장 ─────────────────────────────
    print(f"\n최종 모델 저장 중 ({best_model_name})...")
    tr_f = df[df.date < "2025-01-01"]
    te_f = df[df.date >= "2025-01-01"]
    if len(te_f) < 50:
        n = int(len(df) * 0.1)
        tr_f = df.iloc[:-n]; te_f = df.iloc[-n:]

    if best_model_name == "LightGBM":
        import lightgbm as lgb
        dtrain = lgb.Dataset(tr_f[FEAT_COLS].values.astype(np.float32),
                             label=tr_f["label"].values, feature_name=FEAT_COLS)
        dvalid = lgb.Dataset(te_f[FEAT_COLS].values.astype(np.float32),
                             label=te_f["label"].values, reference=dtrain)
        final = lgb.train(
            LGB_PARAMS, dtrain, num_boost_round=1000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
        )
        val_auc = roc_auc_score(
            te_f["label"].values,
            final.predict(te_f[FEAT_COLS].values.astype(np.float32))
        )
        final.save_model("model_lgbm_us.txt")
        with open("feat_cols_lgbm_us.pkl", "wb") as f:
            pickle.dump(FEAT_COLS, f)

    elif best_model_name == "XGBoost":
        import xgboost as xgb
        dtrain = xgb.DMatrix(tr_f[FEAT_COLS].values.astype(np.float32),
                             label=tr_f["label"].values)
        dvalid = xgb.DMatrix(te_f[FEAT_COLS].values.astype(np.float32),
                             label=te_f["label"].values)
        final = xgb.train(
            XGB_PARAMS, dtrain, num_boost_round=1000,
            evals=[(dvalid, "val")],
            early_stopping_rounds=50, verbose_eval=100
        )
        val_auc = roc_auc_score(
            te_f["label"].values,
            final.predict(xgb.DMatrix(te_f[FEAT_COLS].values.astype(np.float32)))
        )
        final.save_model("model_xgb_us.json")
        with open("feat_cols_lgbm_us.pkl", "wb") as f:
            pickle.dump(FEAT_COLS, f)

    elif best_model_name == "RandomForest":
        final = RandomForestClassifier(
            n_estimators=500, max_depth=8,
            min_samples_leaf=20, n_jobs=-1, random_state=42
        )
        final.fit(tr_f[FEAT_COLS].values.astype(np.float32), tr_f["label"].values)
        pred_val = final.predict_proba(te_f[FEAT_COLS].values.astype(np.float32))[:, 1]
        val_auc  = roc_auc_score(te_f["label"].values, pred_val)
        with open("model_rf_us.pkl", "wb") as f:
            pickle.dump(final, f)
        with open("feat_cols_lgbm_us.pkl", "wb") as f:
            pickle.dump(FEAT_COLS, f)

    print(f"최종 Val AUC: {val_auc:.4f}")
    print(f"저장 완료: {best_model_name}")

    with open("best_model_name.txt", "w") as f:
        f.write(best_model_name)
