#%%
# Y1 - Stage 1 with Final Fold Optimization
import numpy as np, pandas as pd, random, warnings
from dataclasses import dataclass
from typing import List

# Optional GBMs
try: 
    from lightgbm import LGBMRegressor
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False
try: 
    from catboost import Catboostregressor
    HAVE_CAT = True
except Exception:
    HAVE_CAT = False

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbours import NearestNeighbours
from sklearn.pipeline import Pipeline
import inspect

SEED = 1337
random.seed(SEED); np.random.seed(SEED)
warnings.filterwarnings("ignore")

# Loading data
train = pd.read.csv("train.csv")
test = pd.read.csv("test.csv")
time_col = "time"; tgt = "Y1"; id_col = "id" if "id" in test.columns else None
if time_col in train.columns: train = train.sort_values(time_col).reset_index(drop=True)
if time_col in test.columns: test = test.sort_values(time_col).reset_index(drop=True)

# causal features
letters_all = [c for c in train.columns if c.isalpha() and len(c)==1]
letters = [c for c in letters_all if c not in ["O", "P", "Y1", "Y2"] and c in train.columns]
letters = [ c for c in letters if train[c].notna().mean()>0.90]
base_cols = letters + ([time_col] if time_col in train.columns else [])

HL_LIST = [5,10,20,40]
LAG_SET=[1,2,3,5]
EW_ADJ=False

def build_stage1_features(df: pd.DataFrame) -> pd.DataFrame:
    Xb = df[base_cols].copy()
    z={}
    for c in letters:
        s = Xb[c]
        for hl in HL_LIST:
            mu=s.ewm(halflife=hl, adjust=EW_ADJ).mean().shift(1)
            sd=s.ewm(halflife=hl, adjust=EW_ADJ).std().shift(1)
            z[f"{c}_zew_hl{h1}"]=(((s-mu))/sd.replace(0,np.nan)).fillna(0.0)
    Z=pd.DataFrame(z, index=Xb.index)
    lagd = {}
    for c in letters:
        for L in LAG_SET:
            lagd[f"{c}_lag{L}"] = Xb[c].shift(L)
        for hl in HL_LIST:
            zc = f"{c}_zew_hl{hl}"
            if zc in Z.columns:
                for L in LAG_SET:
                    lagd[f"{zc}_lag{L}"] = Z[zc].shift(L)
    LAGS = pd.DataFrame(lagd, index=Xb.index)

    return pd.concat([Xb, Z, LAGS], axis=1)

X_train = build_stage1_features(train)
y = train[tgt].astype(float).values
print("Stage-1 letters:", letters)
print("Stage-1 train feats:", X_train.shape)

# time folds
@dataclass
class Fold:
    tr_idx: np.ndarray
    va_idx: np.ndarray
def make_time_folds(n:int, n_valid:int=4, embargo:int=200)->List[Fold]:
    idx=np.arrange(n); blocks=np.array_split(idx, n_valid+1); folds=[]
    for k in range(1, n_valid+1):
        va=blocks[k]; tr=np.concatenate(blocks[:k])
        if embargo>0: tr=tr[tr < (va[0]-embargo)]
        folds.append(Fold(tr,va))
    return folds
folds = make_time_folds(len(X_train), n_valid=4, embargo=200)
print("Fold sizes:", [(len(f.tr_idx), len(f.va_idx)) for f in folds])
last_va = folds[-1].va_idx
proxy_va = folds[-2].va_idx
VAL_BLOCK = len(last_va)

# KNN stuff
def _select_cols(X, names): return [c for c in names if c in X.columns]
def get_space(space: str, Xtr: pd.DataFrame, Xva: pd.DataFrame):
    if space =="raw_znorm":
        cols=_select_cols(Xtr, letters); 
        if not cols: return None, None
        sc=StandardScaler().fit(Xtr[cols]); return sc.transform(Xtr[cols]), sc.transform(Xva[cols])  
    if space=="zew_multi":
        cols=[f"{c}_zew_hl{hl}" for c in letters for hl in HL_LIST if f"{c}_zew_hl{hl}" in Xtr.columns]
        if not cols: return None, None
        sc=StandardScaler().fit(Xtr[cols]); return sc.transform(Xtr[cols]), sc.transform(Xva[cols])
    if space=="lags_short": 
        cols=[c for c in Xtr.columns if any(c.endswith(f"_lag{L}") for L in LAG_SET)]
        if not cols: return None, None
        sc=StandardScaler().fit(Xtr[cols]); return sc.transform(Xtr[cols]), sc.transform(Xva[cols])
    return None, None

def _row_norm(A):
    n=np.linalg.norm(A, axis=1,keepdims=True); n[n==0]=1.0; return A/n
def knn_predict_numpy(Xtr, ytr, Xva, k=50, metric="cosine", batch=2048):
    XT=np.asarray(Xtr,np.float32); XV=np.asarray(Xva,np.float32); ytr=np.asarray(ytr,np.float64)
    ntr=XT.shape[0]; k=int(min(k,max(1,ntr))); preds=np.empty(XV.shape[0],np.float64)
    if metric=="cosine":
        XTn=_row_norm(XT); XVn=_row_norm(XV)
        for s in range(0, XVn.shape[0], batch):
            e=min(s+batch, XVn.shape[0]); sim=XVn[s:e]@XTn.T; d=1.0-sim
            idx=np.argpartition(d,k-1,axis=1)[:, :k]; di=np.take_along_axis(d, idx, axis=1); yi=ytr[idx]
            sm=np.median(di,axis=1); w=np.ones_like(di); ok=sm>1e-12
            if np.any(ok): w[ok]=np.exp(-(di[ok]/sm[ok,None])**2)
            sw=np.sum(w,axis=1); num=np.sum(w*yi,axis=1); out=np.full(e-s,np.mean(ytr))
            nz=sw>0; out[nz]=num[nz]/sw[nz]; preds[s:e]=out
        return preds
    elif metric=="euclidean":
        xtn2=np.sum(XT*XT,axis=1).astype(np.float64)
        for s in range(0, XV.shape[0], batch):
            e=min(s+batch, XV.shape[0]); Q=XV[s:e]; qn2=np.sum(Q*Q,axis=1,keepdims=True).astype(np.float64)
            D2=np.maximum(0.0, qn2 + xtn2[None,:] - 2.0*(Q@XT.T).astype(np.float64))
            idx=np.argpartition(D2,k-1,axis=1)[:, :k]; di=np.sqrt(np.take_along_axis(D2, idx, axis=1)); yi=ytr[idx]
            sm=np.median(di,axis=1); w=np.ones_like(di); ok=sm>1e-12
            if np.any(ok): w[ok]=np.exp(-(di[ok]/sm[ok,None])**2)
            sw=np.sum(w,axis=1); num=np.sum(w*yi,axis=1); out=np.full(e-s,np.mean(ytr))
            nz=sw>0; out[nz]=num[nz]/sw[nz]; preds[s:e]=out
        return preds
    else:
        raise ValueError("Unsupported metric")
def knn_predict_safe(Xtr, ytr, Xva, k=50, metric="cosine"):
    try:
        k=int(min(k, max(1,Xtr.shape[0]))); nn=NearestNeighbors(n_neighbors=k, metric=metric); nn.fit(Xtr)
        dists, idx = nn.kneighbors(Xva, n_neighbors=k, return_distance=True)
        preds=np.empty(Xva.shape[0], float)
        for i in range(Xva.shape[0]):
            di=dists[i]; yi=ytr[idx[i]]; s=np.median(di)
            w=np.ones_like(di) if not np.isfinite(s) or s<=1e-12 else np.exp(-(di/s)**2)
            sw=np.sum(w); preds[i]=float(np.sum(w*yi)/sw) if sw>0 else float(np.mean(yi))
        return preds
    except Exception:
        return knn_predict_numpy(Xtr,ytr,Xva,k=k,metric=metric,batch=2048)
    
# Fitting models
def s1_model(name):
    if name=="ridge": return Pipeline([("scaler",StandardScaler()),("ridge",Ridge(alpha=1.2,random_state=SEED))])
    if name=="pcr":   return Pipeline([("scaler",StandardScaler()),
                                       ("pca",PCA(n_components=32,random_state=SEED)),
                                       ("ridge",Ridge(alpha=1.0,random_state=SEED))])
    if name=="pls":   return Pipeline([("scaler",StandardScaler()),("pls",PLSRegression(n_components=8))])
    if name=="lgbm" and HAVE_LGBM:
        return LGBMRegressor(n_estimators=2500,learning_rate=0.05,num_leaves=63,
                             min_data_in_leaf=100,subsample=0.9,colsample_bytree=0.8,
                             reg_lambda=5.0,random_state=SEED,objective="regression",
                             force_col_wise=True)
    if name=="cat" and HAVE_CAT:
        return CatBoostRegressor(iterations=2500,depth=6,learning_rate=0.05,l2_leaf_reg=8.0,
                                 loss_function="RMSE",random_seed=SEED,verbose=False)
    return None

def estimator_supports_sample_weight(est) -> bool:
    try:
        sig = inspect.signature(est.fit)
        return "sample_weight" in sig.parameters
    except Exception:
        return False

def fit_with_weight(mdl, X, y, w):
    if not isinstance(mdl, Pipeline):
        try:
            if estimator_supports_sample_weight(mdl):
                return mdl.fit(X, y, sample_weight=w)
        except TypeError:
            pass
        return mdl.fit(X, y)
    last_name, last_est = mdl.steps[-1]
    if estimator_supports_sample_weight(last_est):
        return mdl.fit(X, y, **{f"{last_name}__sample_weight": w})
    else:
        return mdl.fit(X, y)
    
# NN - Ridge
def time_decay_weights(idx: np.ndarray, half_life: int) -> np.ndarray:
    if len(idx)==0: return np.array([])
    t = idx.astype(float); t1=t.max()
    lam = np.log(2.0)/max(1,half_life)
    w = np.exp(-lam*(t1 - t))
    return np.clip(w, 1e-6, None)

def spike_weights(y, gamma=2.0, beta=1.0):
    rk = pd.Series(np.abs(y)).rank(pct=True).values
    return 1.0 + beta * (rk ** gamma)

def solve_nn_ridge_intercept(P, y, alpha=1e-3, w_row=None, iters=2000, restarts=4, seed=SEED):
    rng = np.random.RandomState(seed)
    P = np.asarray(P, float); y = np.asarray(y, float).ravel()
    n, m = P.shape
    W = np.ones(n, float) if w_row is None else np.maximum(np.asarray(w_row, float).ravel(), 1e-12)
    W /= W.mean()
    mu_y = float((W * y).sum() / W.sum())
    mu_P = (W[:, None] * P).sum(axis=0) / W.sum()
    Pc, yc = P - mu_P, y - mu_y
    def power_L():
        v = rng.rand(m); v /= np.linalg.norm(v) + 1e-12
        A = (Pc.T @ (W[:, None]*Pc)) / n
        for _ in range(25):
            v = A @ v + alpha * v
            v /= np.linalg.norm(v) + 1e-12
        return float(v.T @ (A @ v + alpha * v))
    L = 2.0 * power_L(); step = 1.0/(L + 1e-12)
    best_w, best_loss = None, np.inf
    for _ in range(restarts):
        w = np.abs(rng.randn(m))
        for _ in range(iters):
            r = Pc @ w - yc
            grad = (2.0/n) * (Pc.T @ (W * r)) + 2.0*alpha*w
            w = np.clip(w - step*grad, 0.0, None)
        r = Pc @ w - yc
        loss = float((W * r * r).mean() + alpha * (w @ w))
        if loss < best_loss:
            best_loss, best_w = loss, w.copy()
    b = mu_y - float(mu_P @ best_w)
    return best_w, b

# OOF - Predictions
BASE = ["ridge","pcr","pls"] + (["lgbm"] if HAVE_LGBM else []) + (["cat"] if HAVE_CAT else []) \
       + ["knn_raw","knn_zew","knn_lags"]
oof = {nm: np.full(len(y), np.nan) for nm in BASE}

for fi, fold in enumerate(folds, start=1):
    tr, va = fold.tr_idx, fold.va_idx
    Xtr, Xva = X_train.iloc[tr], X_train.iloc[va]
    ytr, yva = y[tr], y[va]
    tr_mask = ~Xtr.isna().any(axis=1); va_mask = ~Xva.isna().any(axis=1)
    Xtr, ytr = Xtr[tr_mask], ytr[tr_mask]
    Xva, yva = Xva[va_mask], yva[va_mask]
    if len(Xtr)==0 or len(Xva)==0: 
        print(f"[S1] Fold {fi} skipped"); continue

    w_tr = time_decay_weights(Xtr.index.values, half_life=VAL_BLOCK)

    for name in ["ridge","pcr","pls","lgbm","cat"]:
        if name not in BASE: continue
        mdl = s1_model(name)
        if mdl is None: continue
        try:
            if name in ("lgbm","cat"):
                mdl.fit(Xtr, ytr, sample_weight=w_tr)
            else:
                fit_with_weight(mdl, Xtr, ytr, w_tr)
        except TypeError:
            mdl.fit(Xtr, ytr)
        oof[name][va[va_mask]] = np.asarray(mdl.predict(Xva)).ravel()

    Xs, Vs = get_space("raw_znorm", Xtr, Xva)
    if Xs is not None:
        oof["knn_raw"][va[va_mask]]  = knn_predict_safe(Xs, ytr, Vs, k=50, metric="cosine")
    Xs, Vs = get_space("zew_multi", Xtr, Xva)
    if Xs is not None:
        oof["knn_zew"][va[va_mask]]  = knn_predict_safe(Xs, ytr, Vs, k=50, metric="cosine")
    Xs, Vs = get_space("lags_short", Xtr, Xva)
    if Xs is not None:
        oof["knn_lags"][va[va_mask]] = knn_predict_safe(Xs, ytr, Vs, k=25, metric="cosine")


# trimming
keep = []
for nm in BASE:
    p = oof[nm][last_va]
    m = np.isfinite(p) & np.isfinite(y[last_va])
    if m.sum() < 50:
        continue
    yp = p[m]; yt = y[last_va][m]
    if np.std(yp) <= 1e-12:
        continue
    r2_last = r2_score(yt, yp)
    corr_last = np.corrcoef(yt, yp)[0,1]
    if (corr_last is not None) and np.isfinite(corr_last) and (corr_last > 0.0) and (r2_last > -0.02):
        keep.append(nm)
if not keep:
   corrs = []
   for nm in BASE:
       p = oof[nm]; m = np.isfinite(p) & np.isfinite(y)
       if m.sum()>100 and np.std(p[m])>0:
           corrs.append((abs(np.corrcoef(y[m], p[m])[0,1]), nm))
       keep = [nm for _,nm in sorted(corrs, reverse=True)[:3]]

print("Kept bases:", keep)

M_all = np.column_stack([oof[n] for n in keep])
mask_all = ~np.isnan(M_all).any(axis=1)
M = M_all[mask_all]; y_mask = y[mask_all] 

# Blenderrr
w_row_global = time_decay_weights(np.arange(len(y))[mask_all], half_life=VAL_BLOCK) * spike_weights(y_mask, 2.0, 1.0)
w_global, b_global = solve_nn_ridge_intercept(M, y_mask, alpha=1e-3, w_row=w_row_global)

# last-fold-only blender (fit on last fold rows only)
m_last = np.zeros(len(y), bool); m_last[last_va] = True
m_last = m_last & mask_all
M_last = M_all[m_last]; y_last = y[m_last]
w_last, b_last = solve_nn_ridge_intercept(M_last, y_last, alpha=1e-3, w_row=None)

m_proxy = np.zeros(len(y), dtype=bool)
m_proxy[proxy_va] = True
m_proxy = m_proxy & mask_all

M_proxy = M_all[m_proxy]
y_proxy = y[m_proxy]

best_alpha, best_r2 = 0.35, -1e9
for a in alpha_grid:
    w_a = (1.0 - a) * w_global + a * w_last
    b_a = (1.0 - a) * b_global + a * b_last
    yhat_p = M_proxy @ w_a + b_a
    r2p = r2_score(y_proxy, yhat_p) if len(yhat_p) > 0 else -1e9
    if r2p > best_r2:
        best_r2, best_alpha = r2p, a
    
w_alpha = (1.0 - best_alpha) * w_global + best_alpha * w_last
b_alpha = (1.0 - best_alpha) * b_global + best_alpha * b_last
print(f"Tuned α (proxy fold): {best_alpha:.2f}")

y1_oof_base = np.full(len(y), np.nan)
y1_oof_base[mask_all] = M @ w_alpha + b_alpha

# Final fold juice
def calibrate_last_shrunk(y_true, y_pred, va_idx, slope_clip=(0.9, 1.1)):
    m = np.isfinite(y_true[va_idx]) & np.isfinite(y_pred[va_idx])
    yt = y_true[va_idx][m]; yp = y_pred[va_idx][m]
    if len(yp) == 0:
        return 1.0, 0.0
    X = np.column_stack([yp, np.ones_like(yp)])
    a, b = np.linalg.lstsq(X, yt, rcond=None)[0]
    a = float(np.clip(a, slope_clip[0], slope_clip[1]))
    return a, float(b)
a_cal, b_cal = calibrate_last_fold_shrunk(y, y1_oof_base, last_va, slope_clip=(0.9, 1.1))
y1_oof = a_cal * y1_oof_base + b_cal

mask_eval = np.isfinite(y1_oof)
r2_global = r2_score(y[mask_eval], y1_oof[mask_eval])
m_last_eval = np.isfinite(y1_oof[last_va]) & np.isfinite(y[last_va])
r2_last = r2_score(y[last_va][m_last_eval], y1_oof[last_va][m_last_eval])

print(f"\nStage-1 OOF R² (tilted+cal): {r2_global:.6f}")
print(f"Stage-1 last-fold OOF R² (tilted+cal): {r2_last:.6f}")
print(f"Calibration: y' = {a_cal:.4f} * y_hat + {b_cal:.4f}")

# full refit + predictions
def build_stage1_features_concat(train_df, test_df):
    df_all = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    X_all = build_stage1_features(df_all)
    X_all = X_all.ffill().fillna(0.0)
    X_tr = X_all.iloc[:len(train_df)].copy()
    X_te = X_all.iloc[len(train_df):].copy()
    return X_tr, X_te
X_tr_full, X_te_full = build_stage1_features_concat(train, test)

def predict_kept_models(Xtr: pd.DataFrame, ytr: np.ndarray, Xte: pd.DataFrame, keep_names: List[str]):
    preds = {}
    w_full = time_decay_weights(Xtr.index.values, half_life=VAL_BLOCK)
    for name in ["ridge","pcr","pls","lgbm","cat"]:
        if name in keep_names:
            mdl = s1_model(name)
            if mdl is None: 
                continue
            try:
                if name in ("lgbm","cat"):
                    mdl.fit(Xtr, ytr, sample_weight=w_full)
                else:
                    fit_with_weight(mdl, Xtr, ytr, w_full)
            except TypeError:
                mdl.fit(Xtr, ytr)
            preds[name] = np.asarray(mdl.predict(Xte)).ravel()

# knn spaces (maybe useless maybe not)
    if "knn_raw" in keep_names:
        Xs_tr, Xs_te = get_space("raw_znorm", Xtr, Xte)
        if Xs_tr is not None:
            preds["knn_raw"] = knn_predict_safe(Xs_tr, ytr, Xs_te, k=50, metric="cosine")
    if "knn_zew" in keep_names:
        Xz_tr, Xz_te = get_space("zew_multi", Xtr, Xte)
        if Xz_tr is not None:
            preds["knn_zew"] = knn_predict_safe(Xz_tr, ytr, Xz_te, k=50, metric="cosine")
    if "knn_lags" in keep_names:
        Xl_tr, Xl_te = get_space("lags_short", Xtr, Xte)
        if Xl_tr is not None:
            preds["knn_lags"] = knn_predict_safe(Xl_tr, ytr, Xl_te, k=25, metric="cosine")
    return preds

preds_keep = predict_kept_models(X_tr_full, y, X_te_full, keep)
Ptest = np.column_stack([preds_keep[nm] for nm in keep if nm in preds_keep])

for nm in keep:
    if nm not in preds_keep:
        Ptest = np.column_stack([Ptest, np.zeros(len(test))])

y1_test_hat = Ptest @ w_alpha + b_alpha
y1_test_hat = a_cal * y1_test_hat + b_cal

test_ids = test[id.col].values if id_col else np.arrange(len(test))
sub = pd.DataFrame({"id": test_ids, "Y1": y1_test_hat})
sub.to_csv("simonclawy1new.csv", index=False)
print("\nSaved submission to simonclawy1new1.csv")
print(sub.head().to_string(index=False))
# %%
