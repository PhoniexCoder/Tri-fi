#!/usr/bin/env python3
import argparse, json, os, time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = (
    [f"rssi_{i}" for i in [1,2,3]] + 
    [f"motion_{i}" for i in [1,2,3]] + 
    [f"breathing_{i}" for i in [1,2,3]] + 
    ["rssi_mean","rssi_range","motion_max","motion_agree","vote_count"]
)

def add_cross(df):
    r1,r2,r3 = df.rssi_1,df.rssi_2,df.rssi_3
    m1,m2,m3 = df.motion_1,df.motion_2,df.motion_3
    df["rssi_mean"]   = (r1+r2+r3)/3
    df["rssi_range"]  = df[["rssi_1","rssi_2","rssi_3"]].max(axis=1)-df[["rssi_1","rssi_2","rssi_3"]].min(axis=1)
    df["motion_max"]  = df[["motion_1","motion_2","motion_3"]].max(axis=1)
    mm = (m1+m2+m3)/3+1e-6
    df["motion_agree"]= df[["motion_1","motion_2","motion_3"]].min(axis=1)/mm
    df["vote_count"]  = ((m1>0.3).astype(int)+(m2>0.3).astype(int)+(m3>0.3).astype(int))/3.0
    return df

# Normalisation to match the inference script
def norm_rssi(r): return ((r+70)/35.0).clip(0,1)
def norm_mot(m): return (m/20.0).clip(0,1)

def sim(rng,n,sc):
    r={}
    # CALIBRATED TO USER ROOM: Absent is around -55dBm, Inside is around -42dBm
    if sc=="absent":
        for i in [1,2,3]:
            r[f"rssi_{i}"]=rng.normal(-55,3,n).clip(-70,-48) # User's actual noisy "outside" RSSI
            r[f"motion_{i}"]=rng.normal(8,2,n).clip(3,15)     # User's actual "outside" Motion
            r[f"breathing_{i}"]=rng.normal(0,0.1,n).clip(0,1)
        r["label"]=np.zeros(n,dtype=int)
    elif sc=="inside":
        for i in [1,2,3]:
            r[f"rssi_{i}"]=rng.normal(-42,2,n).clip(-48,-35) # User's "inside" RSSI
            r[f"motion_{i}"]=rng.normal(7,2,n).clip(4,12)    # User's "inside" Motion (lower/same as noise)
            r[f"breathing_{i}"]=rng.normal(0.4,0.1,n).clip(0.1,1)
        r["label"]=np.ones(n,dtype=int)
    else: # near node false positives
        dom=int(sc[-1])
        for i in [1,2,3]:
            if i==dom:
                r[f"rssi_{i}"]=rng.normal(-38,2,n).clip(-42,-30)
                r[f"motion_{i}"]=rng.normal(15,3,n).clip(10,20)
            else:
                r[f"rssi_{i}"]=rng.normal(-65,4,n).clip(-75,-55)
                r[f"motion_{i}"]=rng.normal(5,2,n).clip(2,10)
            r[f"breathing_{i}"]=rng.normal(0.1,0.1,n).clip(0,0.5)
        r["label"]=np.zeros(n,dtype=int)
    
    df = pd.DataFrame(r)
    for i in [1,2,3]:
        df[f"rssi_{i}"] = norm_rssi(df[f"rssi_{i}"])
        df[f"motion_{i}"] = norm_mot(df[f"motion_{i}"])
    return df

def train(n_rows,prefix):
    rng=np.random.default_rng(42)
    scenarios=["absent","inside","near_1","near_2","near_3"]
    n=n_rows//len(scenarios)
    frames=[sim(rng,n,s) for s in scenarios]
    df=pd.concat(frames,ignore_index=True)
    df=add_cross(df)
    
    X=df[FEATURE_COLS].values; y=df["label"].values
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    sc=StandardScaler()
    Xtrs=sc.fit_transform(Xtr); Xtes=sc.transform(Xte)
    clf=RandomForestClassifier(n_estimators=400,max_depth=10,class_weight="balanced",random_state=42)
    clf.fit(Xtrs,ytr)
    
    sp=f"{prefix}_scaler.pkl"; mp=f"{prefix}_presence.pkl"; ep=f"{prefix}_meta.json"
    joblib.dump(sc,sp); joblib.dump(clf,mp)
    json.dump({"version":"v4_calibrated","scaler_path":sp,"model_presence":mp},open(ep,"w"),indent=2)
    print(f"✓ CALIBRATED MODEL SAVED: {mp}")

if __name__=="__main__":
    train(10000, "model_triangular")
