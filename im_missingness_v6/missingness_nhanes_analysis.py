"""NHANES flag leakage + PSU-grouped flag attack. Portable paths; writes JSON ledger."""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from _path_bootstrap import nhanes_csv, out_path

warnings.filterwarnings("ignore")
rng = np.random.default_rng(1)
d = pd.read_csv(nhanes_csv())
d = d[d.RIDAGEYR >= 20].copy()
d = d[d.SMQ020.isin([1, 2]) & d.DMDCITZN.isin([1, 2]) & d.DMDEDUC2.isin([1, 2, 3, 4, 5])].copy()
d["R_inc"] = d.INDFMPIR.notna().astype(int)
d["R_alq"] = d.ALQ101.isin([1, 2]).astype(int)
d["noncit"] = (d.DMDCITZN == 2).astype(int)
d["smoke"] = (d.SMQ020 == 1).astype(int)
d["lowedu"] = (d.DMDEDUC2 <= 2).astype(int)
d["ageb"] = pd.cut(d.RIDAGEYR, [19, 29, 39, 49, 59, 69, 80]).astype(str)
d["psu"] = d.SDMVSTRA * 10 + d.SDMVPSU
X = pd.get_dummies(d[["ageb", "RIAGENDR", "RIDRETH1"]].astype(str), drop_first=True).astype(float)
ledger = {
    "n": int(len(d)),
    "income_response": float(d.R_inc.mean()),
    "alcohol_response": float(d.R_alq.mean()),
    "prevalence": {
        "noncit": float(d.noncit.mean()),
        "smoke": float(d.smoke.mean()),
        "lowedu": float(d.lowedu.mean()),
    },
    "single_pattern": [],
    "flag_attack": [],
}
print(
    "n=%d  income response rate %.3f  alcohol-questionnaire response rate %.3f"
    % (len(d), d.R_inc.mean(), d.R_alq.mean())
)


def stats(df, R, Y, Xm):
    g1 = df.loc[df[Y] == 1, R].mean()
    g0 = df.loc[df[Y] == 0, R].mean()
    m = sm.Logit(df[R], sm.add_constant(pd.concat([df[[Y]], Xm.loc[df.index]], axis=1))).fit(disp=0)
    return g1, g0, m.params[Y]


def boot(fn, B=400):
    strata = d.groupby("SDMVSTRA").psu.unique()
    out = []
    for b in range(B):
        parts = []
        for s, ps in strata.items():
            for p in rng.choice(ps, len(ps)):
                parts.append(d[d.psu == p])
        bd = pd.concat(parts, ignore_index=True)
        bX = pd.get_dummies(bd[["ageb", "RIAGENDR", "RIDRETH1"]].astype(str), drop_first=True).astype(float)
        try:
            out.append(fn(bd, bX))
        except Exception:
            pass
    return np.array(out)


Phi = lambda g, gp, e: np.log(
    max(
        (1 + g * (np.exp(e) - 1)) / (1 + gp * (np.exp(e) - 1)),
        (1 - g * (1 - np.exp(-e))) / (1 - gp * (1 - np.exp(-e))),
    )
)
print("\n== Table A: single pattern, single driver ==")
for R in ["R_inc", "R_alq"]:
    for Y in ["noncit", "smoke", "lowedu"]:
        g1, g0, gam = stats(d, R, Y, X)
        bs = boot(lambda bd, bX: stats(bd, R, Y, bX))
        e1 = abs(np.log(g1 / g0))
        e0 = abs(np.log((1 - g1) / (1 - g0)))
        be1 = np.abs(np.log(bs[:, 0] / bs[:, 1]))
        be0 = np.abs(np.log((1 - bs[:, 0]) / (1 - bs[:, 1])))
        lv = [e + max(Phi(g1, g0, e), Phi(g0, g1, e)) for e in (0.1, 0.5, 1.0)]
        row = {
            "R": R,
            "Y": Y,
            "g1": float(g1),
            "g0": float(g0),
            "adj_logOR": float(gam),
            "adj_logOR_ci": np.percentile(bs[:, 2], [2.5, 97.5]).tolist(),
            "eps_resp": float(e1),
            "eps_resp_ci": np.percentile(be1, [2.5, 97.5]).tolist(),
            "eps_nonresp": float(e0),
            "eps_nonresp_ci": np.percentile(be0, [2.5, 97.5]).tolist(),
            "hidden_at_0.1_0.5_1": [float(x) for x in lv],
        }
        ledger["single_pattern"].append(row)
        print(
            "%s %s: g1=%.3f g0=%.3f | adj logOR=%.3f [%.3f,%.3f] | eps(resp)=%.3f [%.3f,%.3f] eps(nonresp)=%.3f [%.3f,%.3f] | hidden %.3f %.3f %.3f"
            % (
                R,
                Y,
                g1,
                g0,
                gam,
                *np.percentile(bs[:, 2], [2.5, 97.5]),
                e1,
                *np.percentile(be1, [2.5, 97.5]),
                e0,
                *np.percentile(be0, [2.5, 97.5]),
                *lv,
            )
        )

print("\n== Table D: flag attack (5-fold CV by PSU) ==")
for Y in ["noncit", "lowedu", "smoke"]:
    y = d[Y].values
    X0 = X.values
    X1 = np.c_[X0, d[["R_inc", "R_alq"]].values]
    p0 = np.zeros(len(d))
    p1 = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(X0, y, d.psu.values):
        p0[te] = LogisticRegression(C=10, max_iter=2000).fit(X0[tr], y[tr]).predict_proba(X0[te])[:, 1]
        p1[te] = LogisticRegression(C=10, max_iter=2000).fit(X1[tr], y[tr]).predict_proba(X1[te])[:, 1]
    nr = d.R_inc.values == 0
    both = (d.R_inc.values == 0) & (d.R_alq.values == 0)
    auc0, auc1 = float(roc_auc_score(y, p0)), float(roc_auc_score(y, p1))
    ll0, ll1 = float(log_loss(y, p0)), float(log_loss(y, p1))
    odds_nr = float(
        np.exp(np.mean(np.log(p1[nr] / (1 - p1[nr])) - np.log(p0[nr] / (1 - p0[nr]))))
    )
    odds_r = float(
        np.exp(np.mean(np.log(p1[~nr] / (1 - p1[~nr])) - np.log(p0[~nr] / (1 - p0[~nr]))))
    )
    ledger["flag_attack"].append(
        {
            "Y": Y,
            "auc_x_only": auc0,
            "auc_x_flags": auc1,
            "logloss_x_only": ll0,
            "logloss_x_flags": ll1,
            "prevalence": float(y.mean()),
            "prev_income_nr": float(y[nr].mean()),
            "prev_both_missing": float(y[both].mean()),
            "n_both_missing": int(both.sum()),
            "odds_mult_income_nr": odds_nr,
            "odds_mult_respondents": odds_r,
        }
    )
    print(
        "%s: AUC x-only %.4f, x+patterns %.4f | log-loss %.4f -> %.4f | odds_nr %.3f"
        % (Y, auc0, auc1, ll0, ll1, odds_nr)
    )

_out = out_path("v4_flag_attack.json")
json.dump(ledger, open(_out, "w"), indent=1)
print("wrote", _out)
