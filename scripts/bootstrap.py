"""Recompute family-adjusted confidence intervals for captioning control experiments."""
import argparse
from pathlib import Path
import numpy as np

def intervals(differences,repetitions=10000,family=16,seed=1234):
    x=np.asarray(differences,dtype=np.float64)
    if x.ndim!=2 or not x.size or not np.isfinite(x).all():raise ValueError("Invalid image-level differences.")
    rng=np.random.default_rng(seed);draws=[]
    for i in range(0,repetitions,64):
        ids=rng.integers(0,len(x),size=(min(64,repetitions-i),len(x)))
        draws.append(x[ids].mean(axis=1))
    q=.05/(2*family)
    return x.mean(axis=0),np.quantile(np.concatenate(draws),[q,1-q],axis=0).T


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("file",type=Path)
    a=p.parse_args()
    with np.load(a.file,allow_pickle=False) as s:
        seed = int(s["seed"]) if "seed" in s else 1234
        family = int(s["family"]) if "family" in s else 16
        pt,ci=intervals(s["differences"], family=family, seed=seed)
        np.testing.assert_allclose(pt,s["expected_point"],rtol=0,atol=1e-10)
        np.testing.assert_allclose(ci,s["expected_ci"],rtol=0,atol=1e-10)
        for n,v,c in zip(s["labels"],pt,ci):print(f"{n}: {v:.3f} [{c[0]:.3f}, {c[1]:.3f}]")


if __name__=="__main__":
    main()

