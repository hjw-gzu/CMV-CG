"""Fixed-reference-IDF CIDEr-D oracle; never used for caption selection."""
from collections import defaultdict
import numpy as np

class FixedCider:
    def __init__(self,tokenized_references):
        from pycocoevalcap.cider.cider_scorer import cook_refs
        self.refs={i:cook_refs(values,n=4) for i,values in tokenized_references.items()}
        self.df=defaultdict(float)
        for refs in self.refs.values():
            for gram in set(g for r in refs for g in r):self.df[gram]+=1
        self.logn=np.log(float(len(self.refs)))
        self.ref_vectors={i:[self.vector(r) for r in refs] for i,refs in self.refs.items()}

    def vector(self,counts):
        vec=[defaultdict(float) for _ in range(4)];norm=[0.]*4;length=0
        for gram,tf in counts.items():
            n=len(gram)-1;v=float(tf)*(self.logn-np.log(max(1.,self.df[gram])))
            vec[n][gram]=v;norm[n]+=v*v
            if n==1:length+=tf
        return vec,[np.sqrt(v) for v in norm],length

    def score(self,i,tokenized_caption):
        from pycocoevalcap.cider.cider_scorer import precook
        v,n,length=self.vector(precook(tokenized_caption,n=4));score=np.zeros(4)
        for vr,nr,lr in self.ref_vectors[i]:
            sim=np.zeros(4)
            for k in range(4):
                for gram,value in v[k].items():sim[k]+=min(value,vr[k][gram])*vr[k][gram]
                if n[k]!=0 and nr[k]!=0:sim[k]/=n[k]*nr[k]
                sim[k]*=np.e**(-float(length-lr)**2/(2.*6.**2))
            score+=sim
        return float(score.mean()/len(self.ref_vectors[i])*10.)

