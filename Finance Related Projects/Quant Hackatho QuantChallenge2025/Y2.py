#%%
import numpy as np, pandas as pd, random, warnings, inspect, matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Tuple
#%%
try:
    from lightgbm import LGBMRegressor
    try:
        from lightgbm import early_stopping as LGBM_ES, log_evaluation as LGBM_LOG
    except Exception:
        LGBM_ES= LGBM_LOG=None
    HAVE_LGBM=True
except Exception:
    HAVE_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAVE_CAT = True
except Exception:
    HAVE_CAT = False

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
device=torch.device('cuda' if torch.cuda.is_available() else "cpu")

SEED=1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

for fname in ["train_new.csv"]:
    try:
        tnew=pd.read_csv(fname)
        add = [c for c in tnew.columns if c not in train.columns]
        if add: train = pd.concat([train, tnew[add]], axis =1)
    except Exception:
        pass

time_col = 'time'
target = 'Y2'

if time_col in train.columns:
    train=train.sort_values(time_col).reset_index(drop=True)
if time_col in test.columns:
    test = test.sort_values(time_col).reset_index(drop=True)

VOL_LETTERS = ['A','B','D','F','I','K','L']
letters=[c for c in VOL_LETTERS if c in train.columns]
letters = [c for c in letters if train[c].notna().mean() > 0.9]
base_feat_cols = letters + ([time_col] if time_col in train.columns else [])
print(f"Stage-1 letters: {letters}")
print(f"Use time feature: {time_col in base_feat_cols}")

HL_LIST=[5,10,20,40]
EW_ADJ = False
LAG_SET=[1,2,3,5]
TOP_Y2=['D','K','B','L','F','A','I']

def build_stage1_features(df: pd.DataFrame) -> pd.DataFrame:
    X_base = df[base_feat_cols].copy()

    z_dict={}
    for c in letters:
        s=X_base[c]
        for hl in HL_LIST:
            mu=s.ewm(halflife=hl, adjust=EW_ADJ).mean().shift(1)
            sd=s.ewm(halflife=hl,adjust=EW_ADJ).std().shift(1)
            z=((s-mu)/sd.replace(0,np.nan)).fillna(0.0)
            z_dict[f"{c}_zew_hl{hl}"]=z
    Z=pd.DataFrame(z_dict,index=X_base.index)

    #short lags for top y2
    lag_dict={}
    for c in TOP_Y2:
        if c in X_base.columns:
            for L in LAG_SET:
                lag_dict[f"{c}_lag{L}"] =X_base[c].shift(L)
        for hl in HL_LIST:
            name=f"{c}_zew_hl{hl}"
            if name in Z.columns:
                for L in LAG_SET:
                    lag_dict[f"{name}_lag{L}"]=Z[name].shift(L)
    LAGS = pd.DataFrame(lag_dict,index=X_base.index)
    return pd.concat([X_base, Z, LAGS], axis=1)

X_train_raw= build_stage1_features(train)
y_train = train[target].astype(float).values
print("Stage-1 train feats:", X_train_raw.shape)

def build_op_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if "O" in df.columns:
        O=df["O"].astype(float)
        cap=np.nanpercentile(O,99.5) if np.isfinite(O).any() else 0.0
        Oc = np.clip(O.fillna(0.0), 0.0, cap)
        out["O_log"]=np.log1p(Oc)
        out["O_sqrt"]=np.sqrt(Oc)
        out["O_isna"]=O.isna().astype(int)
        out["O_present"]=(~O.isna()).astype(int)
    if "P" in df.columns:
        P=df["P"].astype(float)
        out["P_imp"]=P.fillna(0.0)
        out["P_isna"]=P.isna().astype(int)
    return out
OP_all = build_op_features(train)

## Time folds! 

@dataclass
class Fold:
    tr_idx: np.ndarray
    va_idx: np.ndarray
def make_time_folds(n:int, n_valid:int=4,embargo: int=200):
    idx=np.arange(n)
    blocks=np.array_split(idx,n_valid+1) # 1 seed + n_valid
    folds=[]
    for k in range(1,n_valid+1):
        va = blocks[k]
        tr = np.concatenate(blocks[:k],axis=0)
        if embargo >0:
            tr = tr[tr<(va[0]-embargo)]
        folds.append(Fold(tr_idx=tr,va_idx=va))
    return folds

folds = make_time_folds(len(X_train_raw),n_valid=4,embargo=200)
VAL_BLOCK = len(folds[1].va_idx) if len(folds) > 1 else max(1000,len(X_train_raw)//10)
print("Fold sizes:",[(len(f.tr_idx), len(f.va_idx)) for f in folds])

# 4 weighting
def time_decay_weights(idx: np.ndarray,half_life:int)->np.ndarray:
    if len(idx)==0: return np.array([])
    t=idx.astype(float);t1=t.max()
    age=(t1-t)
    lam=np.log(2)/max(1, half_life)
    w=np.exp(-lam*age)
    return np.clip(w, 1e-6,None)
def spike_weights(y,gamma=2,beta=1):
    rk=pd.Series(y).rank(pct=True).values
    return 1.0 + beta * (rk** gamma)
def solve_nn_ridge_intercept(P,y,alpha=1e-3,w_row=None,iters=2000,restarts=4,seed=SEED):
    rng=np.random.RandomState(seed)
    P=np.asarray(P,float); y=np.asarray(y, float).ravel()
    n, m=P.shapeW=np.ones(n,float) if w_row is None else np.maximum(np.asarray(w_row, float).ravel(),1e-12)
    W /= W.mean()

    mu_y=float((W*y).sum()/W.sum())
    mu_P=(W[:,None]*P).sum(axis=0)/W.sum()
    Pc=P-mu_P
    yc = y-mu_y

    def power_L():
        v=rng.rand(m); v/=np.linalg.norm(v) +1e-12
        A = (Pc.T@ (W[:, None]*Pc))/n
        for _ in range(25):
            v=A @v +alpha *v
            v/=(np.linalg.norm(v)+1e-12)
        return float(v.T @(A @v + alpha * v))
    L = 2.0* power_L()
    step=1.0/(L+1e-12)

    best_w, best_lost = None,np.inf

    for _ in range(restarts):
        w = np.abs(rng.randn(m))
        for _ in range(iters):
            r = Pc @ w-yc
            grad = (2/n)*(Pc.T@(W*r))+2*alpha*W
            w=np.clip(w-step*grad,0.0,None)
        r=Pc @ w-yc
        loss=float((W*r*r).mean() + alpha * (w@w))
        if loss < best_loss:
            best_loss, best_w=loss, w.copy()
        
    b=mu_y - float(mu_P @ best_w)
    return best_w, b

def _select_cols(X,names): return [c for c in names if c in X.columns]

def get_space(space: str, Xtr: pd.DataFrame, Xva,*,letters_for_space=None):
    L= letters_for_space or letters
    if space == "raw_znorm":
        cols=_select_cols(Xtr,L)
        if not cols: return None, None
        sc = StandardScaler().fit(Xtr[cols])
        return sc.transform(Xtr[cols]), sc.transform(Xva[cols])
    if space == "zew_hl10":
        cols=[f"{c}_zew_hl10" for c in L if f"{c}_new_hl10" in Xtr.columns]
        if not cols: return None, None
        sc = StandardScaler().fit(Xtr[cols])
        return sc.transform(Xtr[cols]), sc.transform(Xva[cols])
    if space == "zew_multi":
        cols=[f"{c}_zew_hl{hl}" for c in L for hl in [5,10,20,40] if f"{c}_zew_hl{hl}" in Xtr.columns]
        if not cols: return None, None
        sc= StandardScaler().fit(Xtr[cols])
    if space == "pca32_raw":
        cols=_select_cols(Xtr,L)
        if not cols: return None, None
        sc= StandardScaler().fit(Xtr[cols])
        Xtr_s = sc.transform(Xtr[cols]); Xva_s =sc.transform(Xva[cols]) 
        pca=PCA(n_components=min(32,Xtr_s.shape[1]),random_state=SEED).fit(Xtr_s)
        return pca.transform(Xtr_s), pca.transform(Xva_s)
    if space == 'lags_top':
        cols = []
        for c in TOP_Y2:
            for LAG in [1,2,3,5]:
                n1=f"{c}_lag{LAG}"
                if n1 in Xtr.columns: cols.append(n1)
            for hl in [5,10,20,40]:
                for LAG in [1,2,3,5]:
                    n2 = f"{c}_zew_hl{hl}_lag{LAG}"
                    if n2 in Xtr.columns: cols.append(n2)
        if not cols: return None, None
        sc = StandardScaler().fit(Xtr[cols])
        return sc.transform(Xtr[cols]), sc.transform(Xva[cols])
    return None, None

def _row_norm(A):
    n = np.linalg.norm(A, axis=1, keepdims=True); n[n==0]=1.0; return A/n

def knn_predict_numpy(Xtr_space, ytr, Xva_space, k=50, metric="cosine", batch=2048):
    XT = np.asarray(Xtr_space, dtype=np.float32)
    XV = np.asarray(Xva_space, dtype=np.float32)
    ytr = np.asarray(ytr, dtype=np.float64)
    ntr = XT.shape[0]; k = int(min(k, max(1, ntr)))
    preds = np.empty(XV.shape[0], dtype=np.float64)

    if metric == "cosine":
        XTn=_row_norm(XT); XVn=_row_norm(XV)
        for s in range(0, XVn.shape[0], batch):
            e=min(s+batch, XVn.shape[0]); sim=XVn[s:e] @ XTn.T; d=1.0-sim
            idx=np.argpartition(d,k-1,axis=1)[:, :k]
            di=np.take_along_axis(d, idx, axis=1); yi=ytr[idx]
            sm=np.median(di,axis=1); w=np.ones_like(di); ok=sm>1e-12
            if np.any(ok): w[ok]=np.exp(-(di[ok]/sm[ok,None])**2)
            sw=np.sum(w,axis=1); num=np.sum(w*yi,axis=1)
            out=np.full(e-s,np.mean(ytr)); nz=sw>0; out[nz]=num[nz]/sw[nz]; preds[s:e]=out
        return preds
    elif metric == "euclidean":
        xtn2=np.sum(XT*XT,axis=1).astype(np.float64)
        for s in range(0, XV.shape[0], batch):
            e=min(s+batch, XV.shape[0]); Q=XV[s:e]
            qn2=np.sum(Q*Q,axis=1,keepdims=True).astype(np.float64)
            D2=np.maximum(0.0, qn2 + xtn2[None,:] - 2.0*(Q @ XT.T).astype(np.float64))
            idx=np.argpartition(D2,k-1,axis=1)[:, :k]
            di=np.sqrt(np.take_along_axis(D2, idx, axis=1)); yi=ytr[idx]
            sm=np.median(di,axis=1); w=np.ones_like(di); ok=sm>1e-12
            if np.any(ok): w[ok]=np.exp(-(di[ok]/sm[ok,None])**2)
            sw=np.sum(w,axis=1); num=np.sum(w*yi,axis=1)
            out=np.full(e-s,np.mean(ytr)); nz=sw>0; out[nz]=num[nz]/sw[nz]; preds[s:e]=out
        return preds
    else:
        raise ValueError("Unsupported metric")

def knn_predict_safe(Xtr_space, ytr, Xva_space, k=50, metric="cosine"):
    try:
        n_train = Xtr_space.shape[0]; k = int(min(k, max(1, n_train)))
        nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="auto")
        nn.fit(Xtr_space)
        dists, idx = nn.kneighbors(Xva_space, n_neighbors=k, return_distance=True)
        preds = np.empty(Xva_space.shape[0], dtype=float)
        for i in range(Xva_space.shape[0]):
            di = dists[i]; yi = ytr[idx[i]]
            s  = np.median(di)
            w  = np.ones_like(di) if (not np.isfinite(s) or s<=1e-12) else np.exp(- (di/s)**2 )
            sw = np.sum(w)
            preds[i] = float(np.sum(w*yi)/sw) if sw>0 else float(np.mean(yi))
        return preds
    except Exception:
        return knn_predict_numpy(Xtr_space, ytr, Xva_space, k=k, metric=metric)

def knn_name(space, metric, k): return f"knn_{space}_{metric}_k{k}"

# modest Stage-1 / Stage-2 kNN specs
KNN_S1_SPECS = [("raw_znorm", "cosine", 50), ("zew_hl10", "cosine", 50),
                ("zew_multi", "cosine", 50), ("pca32_raw", "cosine", 50)]
KNN_S2_SPECS = [("lags_top", "cosine", 25), ("lags_top", "euclidean", 25)]

#6 stage 1
def build_s1_model(name):
    # Linear family (pipelines where last step might not be 'est')
    if name == "ridge":
        return Pipeline([("scaler", StandardScaler()),
                         ("ridge", Ridge(alpha=1.5, random_state=SEED))])
    if name == "enet":
        return Pipeline([("scaler", StandardScaler()),
                         ("enet", ElasticNet(alpha=0.001, l1_ratio=0.15, random_state=SEED, max_iter=2500))])
    if name == "pls":
        return Pipeline([("scaler", StandardScaler()),
                         ("pls", PLSRegression(n_components=8))])  # PLS doesn't support sample_weight
    if name == "pcr":
        return Pipeline([("scaler", StandardScaler()),
                         ("pca", PCA(n_components=32, random_state=SEED)),
                         ("ridge", Ridge(alpha=1.0, random_state=SEED))])

    # GBMs (O/P only here)
    if name == "lgbm" and HAVE_LGBM:
        return LGBMRegressor(
            n_estimators=4000, learning_rate=0.04, num_leaves=63,
            min_data_in_leaf=80, subsample=0.85, colsample_bytree=0.8,
            reg_lambda=5.0, random_state=SEED, objective="regression",
            force_col_wise=True
        )
    if name == "cat" and HAVE_CAT:
        return CatBoostRegressor(
            iterations=3000, depth=7, learning_rate=0.04, l2_leaf_reg=8.0,
            loss_function="RMSE", random_seed=SEED, verbose=False
        )
    return None

def estimator_supports_sample_weight(est) -> bool:
    try:
        sig = inspect.signature(est.fit)
        return "sample_weight" in sig.parameters
    except Exception:
        return False

def fit_with_weight(mdl, X, y, w):
    """
    Fit mdl passing sample_weight only if supported.
    Works for plain estimators and Pipelines (routes to last step).
    """
    # Cat/LGBM etc (not Pipeline)
    if not isinstance(mdl, Pipeline):
        try:
            if estimator_supports_sample_weight(mdl):
                return mdl.fit(X, y, sample_weight=w)
        except TypeError:
            pass
        return mdl.fit(X, y)

    # Pipeline: detect last step name + estimator
    last_name, last_est = mdl.steps[-1]
    if estimator_supports_sample_weight(last_est):
        return mdl.fit(X, y, **{f"{last_name}__sample_weight": w})
    else:
        return mdl.fit(X, y)

BASE_S1_MODELS = ["ridge","enet","pls","pcr","lgbm","cat"]
BASE_S1 = BASE_S1_MODELS + [knn_name(s,m,k) for (s,m,k) in KNN_S1_SPECS]

oof_s1 = {name: np.full(len(y_train), np.nan) for name in BASE_S1}
models_s1 = defaultdict(list)

HALF_LIFE = VAL_BLOCK  # gentle recency emphasis

for f_i, f in enumerate(folds, start=1):
    tr, va = f.tr_idx, f.va_idx
    Xtr_lin, Xva_lin = X_train_raw.iloc[tr].copy(), X_train_raw.iloc[va].copy()
    ytr, yva = y_train[tr], y_train[va]

    # O/P augment for GBMs (imputed + masks)
    gbm_cols = [c for c in ["O_log","O_sqrt","O_isna","O_present","P_imp","P_isna"] if c in OP_all.columns]
    Xtr_gbm = pd.concat([Xtr_lin, OP_all.loc[Xtr_lin.index, gbm_cols]], axis=1) if gbm_cols else Xtr_lin
    Xva_gbm = pd.concat([Xva_lin, OP_all.loc[Xva_lin.index, gbm_cols]], axis=1) if gbm_cols else Xva_lin

    # causal masks
    tr_mask_lin = ~Xtr_lin.isna().any(axis=1)
    va_mask_lin = ~Xva_lin.isna().any(axis=1)
    tr_mask_gbm = ~Xtr_gbm.isna().any(axis=1)
    va_mask_gbm = ~Xva_gbm.isna().any(axis=1)

    if not (tr_mask_lin.any() and va_mask_lin.any()):
        print(f"[S1] Fold {f_i} skipped (NaNs)"); continue
    print(f"[S1] Fold {f_i} tr/va (lin): {Xtr_lin[tr_mask_lin].shape} / {Xva_lin[va_mask_lin].shape}")
    if gbm_cols:
        print(f"     Fold {f_i} tr/va (gbm+O/P): {Xtr_gbm[tr_mask_gbm].shape} / {Xva_gbm[va_mask_gbm].shape}")

    # time-decay weights on training rows (by absolute indices)
    wdec_lin = time_decay_weights(Xtr_lin[tr_mask_lin].index.values, half_life=HALF_LIFE)
    wdec_gbm = time_decay_weights(Xtr_gbm[tr_mask_gbm].index.values, half_life=HALF_LIFE)

    # param models
    for name in BASE_S1_MODELS:
        mdl = build_s1_model(name)
        if mdl is None:
            oof_s1[name][va[va_mask_lin]] = np.nan
            models_s1[name].append(None)
            continue
        if name in ("lgbm","cat"):
            Xtr_m = Xtr_gbm[tr_mask_gbm]; ytr_m = ytr[tr_mask_gbm]
            Xva_m = Xva_gbm[va_mask_gbm]; yva_m = yva[va_mask_gbm]
            try:
                if name == "lgbm":
                    cbs = []
                    if LGBM_ES is not None:  cbs.append(LGBM_ES(200))
                    if LGBM_LOG is not None: cbs.append(LGBM_LOG(0))
                    mdl.fit(Xtr_m, ytr_m, sample_weight=wdec_gbm,
                            eval_set=[(Xva_m, yva_m)], eval_metric="l2", callbacks=cbs)
                else:
                    mdl.fit(Xtr_m, ytr_m, sample_weight=wdec_gbm,
                            eval_set=[(Xva_m, yva_m)], use_best_model=True, verbose=False)
            except TypeError:
                mdl.fit(Xtr_m, ytr_m)
            p = np.asarray(mdl.predict(Xva_m)).ravel()
            oof_s1[name][va[va_mask_gbm]] = p
        else:
            Xtr_m = Xtr_lin[tr_mask_lin]; ytr_m = ytr[tr_mask_lin]
            Xva_m = Xva_lin[va_mask_lin]; yva_m = yva[va_mask_lin]
            fit_with_weight(mdl, Xtr_m, ytr_m, wdec_lin)  # <<< FIXED
            p = np.asarray(mdl.predict(Xva_m)).ravel()
            oof_s1[name][va[va_mask_lin]] = p
        models_s1[name].append(mdl)

    # kNN bases (unchanged spaces; no sample_weight)
    for (space, metric, k) in KNN_S1_SPECS:
        nm = knn_name(space, metric, k)
        Xtr_s, Xva_s = get_space(space, Xtr_lin[tr_mask_lin], Xva_lin[va_mask_lin], letters_for_space=letters)
        if Xtr_s is None:
            oof_s1[nm][va[va_mask_lin]] = np.nan
            continue
        p_knn = knn_predict_safe(Xtr_s, ytr[tr_mask_lin], Xva_s, k=k, metric=metric)
        oof_s1[nm][va[va_mask_lin]] = p_knn

# Stage-1 blend — non-neg ridge with intercept + time-decay row weights
M1 = np.column_stack([oof_s1[n] for n in BASE_S1])
mask1 = ~np.isnan(M1).any(axis=1)
w_row_s1 = time_decay_weights(np.arange(len(y_train))[mask1], half_life=HALF_LIFE)
w1, b1 = solve_nn_ridge_intercept(M1[mask1], y_train[mask1], alpha=1e-3, w_row=w_row_s1)
y_s1_oof = np.full(len(y_train), np.nan); y_s1_oof[mask1] = M1[mask1] @ w1 + b1
print("\n[S1] OOF R²:", r2_score(y_train[mask1], y_s1_oof[mask1]))
print("[S1] Weights (first 10):", dict(zip(BASE_S1[:10], np.round(w1[:10],4))), " | b=", round(b1,4))

#7 stage 2 residuals training (HAR(|Y2|) + Y1 leverage/HV + ACCEL + O-dyn + LSTM)
residual_all = y_train - y_s1_oof  # NaN where S1 missing

Y1 = train["Y1"].astype(float) if "Y1" in train.columns else pd.Series(np.nan, index=train.index)
absY2 = np.abs(train[target].astype(float))

def har_feat(s, wins=(5,20,60,120)):
    out={}
    for w in wins:
        out[f"har_mean_{w}"] = s.rolling(w, min_periods=max(3, w//2)).mean().shift(1)
    for w in (20,60):
        out[f"har_max_{w}"]  = s.rolling(w, min_periods=max(5,w//3)).max().shift(1)
        out[f"har_q95_{w}"]  = s.rolling(w, min_periods=max(5,w//3)).quantile(0.95).shift(1)
    return pd.DataFrame(out)
HAR_Y2 = har_feat(absY2)

def leverage_block(y1):
    out={}
    for L in [1,2,3,5,10]:
        out[f"absY1_lag{L}"] = y1.abs().shift(L)
        out[f"Y1sq_lag{L}"]  = (y1**2).shift(L)
        out[f"lev_lag{L}"]   = ((y1<0).astype(int)*(y1**2)).shift(L)
    r_dec = y1/100.0
    if np.any(1.0 + r_dec <= 0): r_dec = y1/10000.0
    logr = np.log1p(r_dec)*100.0
    for w in [5,20,60]:
        mu = logr.rolling(w, min_periods=max(3,w//2)).mean()
        hv = np.sqrt(((logr - mu)**2).rolling(w, min_periods=max(3,w//2)).mean()).shift(1)
        out[f"hvY1_{w}"] = hv
    for w in [20,60]:
        out[f"vov_absY1_{w}"] = y1.abs().rolling(w, min_periods=max(5,w//3)).std().shift(1)
    return pd.DataFrame(out)
LEV_Y1 = leverage_block(Y1)

def build_accel(X_base: pd.DataFrame):
    zcols = [c for c in X_base.columns if "_zew_hl" in c]
    Z = X_base[zcols]
    out={}
    for c in TOP_Y2:
        c5, c10, c20, c40 = f"{c}_zew_hl5", f"{c}_zew_hl10", f"{c}_zew_hl20", f"{c}_zew_hl40"
        if all(x in Z.columns for x in [c5,c20]):  out[f"{c}_sls_5_20"]  = (Z[c5] - Z[c20])
        if all(x in Z.columns for x in [c10,c40]): out[f"{c}_sls_10_40"] = (Z[c10] - Z[c40])
        for name in [c5,c10,c20,c40]:
            if name in Z.columns: out[f"{name}_diff1"] = Z[name].diff(1).fillna(0.0)
    return pd.DataFrame(out, index=X_base.index)
ACCEL = build_accel(X_train_raw)

def build_O_dynamics(OP: pd.DataFrame):
    out = pd.DataFrame(index=OP.index)
    if "O_log" in OP.columns:
        Olog = OP["O_log"].astype(float).fillna(0.0)
        for L in [1,2,3,5]:
            out[f"Olog_lag{L}"] = Olog.shift(L).fillna(0.0)
        dO  = Olog.diff(1).shift(1).fillna(0.0)
        ddO = dO.diff(1).shift(1).fillna(0.0)
        out["O_d1"]  = dO; out["O_d2"] = ddO
        for hl in [5,10]:
            mu = Olog.ewm(halflife=hl, adjust=False).mean()
            sd = Olog.ewm(halflife=hl, adjust=False).std()
            z  = ((Olog - mu) / sd.replace(0, np.nan)).shift(1).fillna(0.0)
            out[f"O_zew_hl{hl}"] = z
        if "O_present" in OP.columns: out["O_present"] = OP["O_present"].astype(int)
        if "O_isna" in OP.columns:    out["O_isna"]    = OP["O_isna"].astype(int)
    if "P_imp" in OP.columns:  out["P_imp"]  = OP["P_imp"].astype(float).fillna(0.0)
    if "P_isna" in OP.columns: out["P_isna"] = OP["P_isna"].astype(int)
    if "har_mean_5" in HAR_Y2.columns and "Olog_lag1" in out.columns:
        out["Olog_lag1_x_har5"] = out["Olog_lag1"] * HAR_Y2["har_mean_5"].fillna(0.0)
    return out
O_DYN = build_O_dynamics(OP_all)

lag_feature_cols = [c for c in X_train_raw.columns if "_lag" in c]
S2_EXTRA = pd.concat([HAR_Y2, LEV_Y1, ACCEL, O_DYN], axis=1)

def build_res_model(name):
    if name == "lgbm_res" and HAVE_LGBM:
        return LGBMRegressor(
            n_estimators=2500, learning_rate=0.05, num_leaves=63,
            min_data_in_leaf=100, subsample=0.9, colsample_bytree=0.8,
            reg_lambda=5.0, random_state=SEED, objective="regression",
            force_col_wise=True
        )
    if name == "cat_res" and HAVE_CAT:
        return CatBoostRegressor(
            iterations=2000, depth=6, learning_rate=0.05, l2_leaf_reg=8.0,
            loss_function="RMSE", random_seed=SEED, verbose=False
        )
    return None

BASE_S2 = []
if HAVE_LGBM: BASE_S2.append("lgbm_res")
if HAVE_CAT:  BASE_S2.append("cat_res")
BASE_S2 += [knn_name(s,m,k) for (s,m,k) in KNN_S2_SPECS] + ["lstm_res"]
oof_s2 = {nm: np.full(len(y_train), np.nan) for nm in BASE_S2}

# LSTM residual model
class LSTMRes(nn.Module):
    def __init__(self, input_dim, hidden=96, layers=2, drop=0.30):
        super().__init__()
        self.h, self.layers = hidden, layers
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self, x):
        B = x.size(0)
        h0 = torch.zeros(self.layers, B, self.h, device=x.device)
        c0 = torch.zeros(self.layers, B, self.h, device=x.device)
        out,_ = self.lstm(x, (h0,c0))
        out = self.drop(out[:, -1, :])
        return self.fc(out).squeeze(-1)

def build_sequences_train(F_tr, y_tr, window):
    Xs, ys, idx = [], [], []
    Z = F_tr.values; yt = y_tr.values
    for t in range(window, len(F_tr)):
        block = Z[t-window:t]
        if np.any(~np.isfinite(block)) or not np.isfinite(yt[t]): continue
        Xs.append(block); ys.append(yt[t]); idx.append(F_tr.index[t])
    X = torch.tensor(np.array(Xs), dtype=torch.float32)
    y = torch.tensor(np.array(ys), dtype=torch.float32)
    return X, y, np.array(idx)

def build_sequences_val(F_tr_tail, F_va, y_va, window):
    F = pd.concat([F_tr_tail.tail(window), F_va], axis=0)
    Xs, ys, idx = [], [], []
    Z = F.values
    for i in range(window, len(F)):
        idx_i = F.index[i]
        if idx_i not in y_va.index: continue
        block = Z[i-window:i]
        if np.any(~np.isfinite(block)) or not np.isfinite(y_va.loc[idx_i]): continue
        Xs.append(block); ys.append(y_va.loc[idx_i]); idx.append(idx_i)
    X = torch.tensor(np.array(Xs), dtype=torch.float32)
    y = torch.tensor(np.array(ys), dtype=torch.float32)
    return X, y, np.array(idx)

# LSTM hparams
win = 128; batch = 256; epochs = 40; patience = 6
lr = 1e-3; wd = 1e-5

for f_i, f in enumerate(folds, start=1):
    tr, va = f.tr_idx, f.va_idx

    # Base lag features + extras (all causal; imputed already)
    Xtr_b = X_train_raw.iloc[tr][lag_feature_cols].copy()
    Xva_b = X_train_raw.iloc[va][lag_feature_cols].copy()
    Xtr_b = pd.concat([Xtr_b, S2_EXTRA.iloc[tr]], axis=1)
    Xva_b = pd.concat([Xva_b, S2_EXTRA.iloc[va]], axis=1)

    ytr_r = residual_all[tr]
    yva_r = residual_all[va]

    tr_mask = (~Xtr_b.isna().any(axis=1)) & np.isfinite(ytr_r)
    va_mask = (~Xva_b.isna().any(axis=1)) & np.isfinite(yva_r)

    Xtr_b, ytr_r = Xtr_b[tr_mask], ytr_r[tr_mask]
    Xva_b, yva_r = Xva_b[va_mask], yva_r[va_mask]
    if len(Xtr_b)==0 or len(Xva_b)==0:
        print(f"[S2] Fold {f_i} skipped (NaNs)"); continue
    print(f"[S2] Fold {f_i} tr/va:", Xtr_b.shape, Xva_b.shape)

    # time-decay weights for residual fits
    wdec_res = time_decay_weights(Xtr_b.index.values, half_life=VAL_BLOCK)

    # Param residual models
    for name in ["lgbm_res","cat_res"]:
        if name not in BASE_S2: continue
        mdl = build_res_model(name)
        if mdl is None:
            oof_s2[name][va[va_mask]] = np.nan
            continue
        try:
            if name == "lgbm_res":
                cbs = []
                if LGBM_ES is not None:  cbs.append(LGBM_ES(200))
                if LGBM_LOG is not None: cbs.append(LGBM_LOG(0))
                mdl.fit(Xtr_b, ytr_r, sample_weight=wdec_res,
                        eval_set=[(Xva_b, yva_r)], eval_metric="l2", callbacks=cbs)
            else:
                mdl.fit(Xtr_b, ytr_r, sample_weight=wdec_res,
                        eval_set=[(Xva_b, yva_r)], use_best_model=True, verbose=False)
        except TypeError:
            mdl.fit(Xtr_b, ytr_r)
        p = np.asarray(mdl.predict(Xva_b)).ravel()
        oof_s2[name][va[va_mask]] = p

    # kNN residuals (lag space)
    for (space, metric, k) in KNN_S2_SPECS:
        nm = knn_name(space, metric, k)
        Xtr_s, Xva_s = get_space(space, Xtr_b, Xva_b, letters_for_space=letters)
        if Xtr_s is None:
            oof_s2[nm][va[va_mask]] = np.nan
            continue
        p_knn = knn_predict_safe(Xtr_s, ytr_r, Xva_s, k=k, metric=metric)
        oof_s2[nm][va[va_mask]] = p_knn

    # LSTM residual
    F_tr = Xtr_b.copy(); F_va = Xva_b.copy()
    F_tr["Y2_inp"] = train.loc[F_tr.index, target].values
    F_va["Y2_inp"] = train.loc[F_va.index, target].values

    sc = StandardScaler()
    F_tr_z = pd.DataFrame(sc.fit_transform(F_tr), index=F_tr.index, columns=F_tr.columns)
    F_va_z = pd.DataFrame(sc.transform(F_va),   index=F_va.index, columns=F_va.columns)

    Xtr_seq, ytr_seq, _         = build_sequences_train(F_tr_z, pd.Series(ytr_r, index=F_tr_z.index), win)
    Xva_seq, yva_seq, idx_vaseq = build_sequences_val(F_tr_z, F_va_z, pd.Series(yva_r, index=F_va_z.index), win)

    if len(Xtr_seq)>0 and len(Xva_seq)>0:
        train_loader = DataLoader(TensorDataset(Xtr_seq, ytr_seq), batch_size=batch, shuffle=True)
        val_loader   = DataLoader(TensorDataset(Xva_seq, yva_seq), batch_size=batch, shuffle=False)

        model = LSTMRes(input_dim=F_tr_z.shape[1], hidden=96, layers=2, drop=0.30).to(device)
        opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        lossf = nn.MSELoss()
        best_loss, best_state, noimp = np.inf, None, 0

        for ep in range(40):
            model.train(); tr_loss=0.0
            for xb,yb in train_loader:
                xb=xb.to(device); yb=yb.to(device)
                opt.zero_grad(); pred=model(xb); loss=lossf(pred,yb)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); tr_loss += loss.item()*len(xb)
            tr_loss /= len(train_loader.dataset)

            model.eval(); va_loss=0.0
            with torch.no_grad():
                for xb,yb in val_loader:
                    va_loss += lossf(model(xb.to(device)), yb.to(device)).item()*len(xb)
            va_loss /= len(val_loader.dataset)

            if va_loss < best_loss - 1e-6:
                best_loss = va_loss
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                noimp=0
            else:
                noimp+=1
                if noimp>=6: break

        if best_state is not None: model.load_state_dict(best_state)
        model.eval()
        preds=[]
        with torch.no_grad():
            for xb,_ in val_loader:
                preds.append(model(xb.to(device)).cpu().numpy())
        preds = np.concatenate(preds)
        oof_s2["lstm_res"][idx_vaseq] = preds
    else:
        oof_s2["lstm_res"][va[va_mask]] = np.nan

#  Stage-2 blend — NN-Ridge(+b), spike *and* time-decay weights
M2 = np.column_stack([oof_s2[n] for n in BASE_S2])
mask2 = ~np.isnan(M2).any(axis=1) & np.isfinite(residual_all)
w_row_s2 = spike_weights(np.abs(y_train[mask2]), gamma=2.0, beta=1.0) * \
           time_decay_weights(np.arange(len(y_train))[mask2], half_life=VAL_BLOCK)
w2, b2 = solve_nn_ridge_intercept(M2[mask2], residual_all[mask2], alpha=1e-3, w_row=w_row_s2)
r_oof = np.full(len(y_train), np.nan); r_oof[mask2] = M2[mask2] @ w2 + b2

print("\n[S2] OOF R² vs residual:", r2_score(residual_all[mask2], r_oof[mask2]))
print("[S2] Weights:", dict(zip(BASE_S2, np.round(w2,4))), " | b=", round(b2,4))

#8 combine and reporting

y2_oof = y_s1_oof + r_oof
mask_all = np.isfinite(y2_oof)
oof_r2_global = r2_score(y_train[mask_all], y2_oof[mask_all])
last_va = folds[-1].va_idx
mask_last = np.isfinite(y2_oof[last_va])
oof_r2_last = r2_score(y_train[last_va][mask_last], y2_oof[last_va][mask_last])

print(f"\n[Combined] Global OOF R²: {oof_r2_global:.6f}")
print(f"[Combined] Last-fold OOF R²: {oof_r2_last:.6f}")

# Tail overlay
tail = 4000
ts = pd.DataFrame({"Y2": y_train, "OOF": y2_oof}).iloc[-tail:]
ts = ts[np.isfinite(ts["Y2"]) & np.isfinite(ts["OOF"])]
plt.figure(figsize=(12,4))
plt.plot(ts.index, ts["Y2"], label="Y2", linewidth=1.0)
plt.plot(ts.index, ts["OOF"], label="Stage-1 NN-Ridge+b (decay) + Stage-2 (HAR/LEV/ACCEL + O-dyn + LSTM)", linewidth=1.2)
plt.title("Y2 — OOF vs truth (last segment)")
plt.xlabel("row"); plt.ylabel("value"); plt.legend(); plt.tight_layout(); plt.show()

# 9.1 If present, append any extra test columns (e.g., O/P) from test_new.csv
try:
    tnew = pd.read_csv("test_new.csv")
    add = [c for c in tnew.columns if c not in test.columns]
    if add:
        test = pd.concat([test, tnew[add]], axis=1)
        print(f"Appended to test: {add}")
except Exception:
    pass

n_train = len(train)
n_test  = len(test)

# 9.2 Build Stage-1 features on TRAIN+TEST together (causal lags for test head)
df_all    = pd.concat([train, test], ignore_index=True)
X_all_raw = build_stage1_features(df_all)

X_test_raw = X_all_raw.iloc[n_train:].copy()
X_test_raw.index = test.index  # align indices with test

# 9.3 O/P features for GBMs on test
OP_test = build_op_features(test)

# 9.4 Fit compact "final" Stage-1 models on full train and predict test
#     (We still use your learned NN-Ridge blend weights w1,b1)
p_s1_test = {name: np.full(n_test, np.nan, dtype=float) for name in BASE_S1}

mask_lin_full = ~X_train_raw.isna().any(axis=1)
wdec_full = time_decay_weights(np.arange(n_train)[mask_lin_full], half_life=HALF_LIFE)

# Parametric S1 models
for name in BASE_S1_MODELS:
    mdl = build_s1_model(name)
    if mdl is None:
        continue

    if name in ("lgbm", "cat"):
        gbm_cols = [c for c in ["O_log","O_sqrt","O_isna","O_present","P_imp","P_isna"] if c in OP_all.columns]
        Xtr = pd.concat([X_train_raw.loc[mask_lin_full], OP_all.loc[mask_lin_full, gbm_cols]], axis=1) if gbm_cols else X_train_raw.loc[mask_lin_full]
        Xte = pd.concat([X_test_raw, OP_test[gbm_cols]], axis=1) if gbm_cols else X_test_raw

        try:
            if name == "lgbm":
                cbs = []
                if LGBM_ES is not None:  cbs.append(LGBM_ES(200))
                if LGBM_LOG is not None: cbs.append(LGBM_LOG(0))
                mdl.fit(Xtr, y_train[mask_lin_full], sample_weight=wdec_full,
                        eval_set=[(Xtr, y_train[mask_lin_full])], callbacks=cbs)
            else:
                mdl.fit(Xtr, y_train[mask_lin_full], sample_weight=wdec_full,
                        eval_set=[(Xtr, y_train[mask_lin_full])], use_best_model=True, verbose=False)
        except TypeError:
            mdl.fit(Xtr, y_train[mask_lin_full])

        p_s1_test[name] = np.asarray(mdl.predict(Xte)).ravel()

    else:
        Xtr = X_train_raw.loc[mask_lin_full]
        Xte = X_test_raw
        mdl = fit_with_weight(mdl, Xtr, y_train[mask_lin_full], wdec_full)
        p_s1_test[name] = np.asarray(mdl.predict(Xte)).ravel()

# kNN S1 bases
for (space, metric, k) in KNN_S1_SPECS:
    nm = knn_name(space, metric, k)
    Xtr_s, Xte_s = get_space(space, X_train_raw.loc[mask_lin_full], X_test_raw, letters_for_space=letters)
    if Xtr_s is None:
        continue
    p_s1_test[nm] = knn_predict_safe(Xtr_s, y_train[mask_lin_full], Xte_s, k=k, metric=metric)

# Assemble S1 test matrix in BASE_S1 order; impute any NaNs with column means
M1_test = np.column_stack([p_s1_test.get(n, np.full(n_test, np.nan)) for n in BASE_S1])
col_means = np.nanmean(M1_test, axis=0)
nan_rows, nan_cols = np.where(np.isnan(M1_test))
if len(nan_rows):
    M1_test[nan_rows, nan_cols] = np.take(col_means, nan_cols)
y_s1_test = M1_test @ w1 + b1

# 9.5 Stage-2 residuals on test (use available KNN residual specs; leave others as 0)
lag_feature_cols = [c for c in X_train_raw.columns if "_lag" in c]
Xtr_b = X_train_raw[lag_feature_cols]
Xte_b = X_test_raw[lag_feature_cols]

p_s2_test = {nm: np.zeros(n_test, dtype=float) for nm in BASE_S2}  # default 0s
mask_res = np.isfinite(residual_all)
Xtr_b_res = Xtr_b.loc[mask_res]
ytr_r_res = residual_all[mask_res]

for (space, metric, k) in KNN_S2_SPECS:
    nm = knn_name(space, metric, k)
    Xtr_s, Xte_s = get_space(space, Xtr_b_res, Xte_b, letters_for_space=letters)
    if Xtr_s is None:
        continue
    p_s2_test[nm] = knn_predict_safe(Xtr_s, ytr_r_res, Xte_s, k=k, metric=metric)

M2_test = np.column_stack([p_s2_test[n] for n in BASE_S2])
r_test  = M2_test @ w2 + b2

# 9.6 Final Y2 predictions + CSV
y2_test_pred = (y_s1_test + r_test).astype(float)
if not np.isfinite(y2_test_pred).all():
    y2_test_pred = np.where(np.isfinite(y2_test_pred), y2_test_pred, np.nanmean(y2_test_pred))

submission = pd.DataFrame({"Y2": y2_test_pred})
submission.to_csv("submission_y2.csv", index=False)
print("\nWrote test predictions to submission_y2.csv")
# %%
