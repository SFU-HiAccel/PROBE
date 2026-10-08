"""ScaleDoc's encoder/loss/augmentation on purchased training labels only.

The caller selects and labels the training subset before entering this module.
No held-out population labels are accepted. Architectural/loss code is imported
from the local upstream copy. The training loop avoids its accumulation of live
autograd graphs and its fixed global checkpoint filenames.
"""
from pathlib import Path
import importlib.util
import hashlib
import time
import numpy as np

import os
import sys
def source_path():
    return Path(os.environ['PROBE_SCALEDOC_SOURCE']).resolve()/'src'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def train_scores(doc_embeddings,query_embedding,training_ids,training_labels,*,seed=0,
                 epochs=(50,30),batch_size=128):
    """Return cosine scores or an explicit inability to train; never fabricate labels.

    Phase 1 starts phase 2 from its best validation checkpoint. Following the
    original train_cl.py, inference uses phase 2's final weights. The original
    aug_split uses zero-noise duplication in its active branch despite the paper
    describing Gaussian perturbations; that implementation is preserved/reported.
    """
    ids=np.asarray(training_ids,dtype=np.int64)
    source=source_path()
    labels=np.asarray(training_labels)
    docs=np.asarray(doc_embeddings,dtype=np.float32)
    query=np.asarray(query_embedding,dtype=np.float32).reshape(-1)
    if len(ids)!=len(labels) or len(set(ids.tolist()))!=len(ids):
        raise ValueError('Training IDs must be unique and aligned')
    if not np.isin(labels,[0,1]).all():raise ValueError('Only purchased binary labels may train')
    if ids.size and (ids.min()<0 or ids.max()>=len(docs)):raise ValueError('Training ID out of range')
    if docs.ndim!=2 or docs.shape[1]!=len(query):raise ValueError('Embedding dimensions differ')
    if len(epochs)!=2 or min(epochs)<1:raise ValueError('Two positive epoch counts required')
    counts=np.bincount(labels.astype(int),minlength=2)
    metadata=dict(seed=seed,epochs=list(epochs),training_size=len(ids),class_counts=counts.tolist(),
        source_hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                       for name in ['model.py','utils/sample.py','train_cl.py','utils/train.py']},
        augmentation='upstream aug_split; inspect recorded upstream source hashes',
        checkpoint_policy='best validation phase1; final phase2 as upstream train_cl.py')
    # Upstream's rebalancer takes half the minority for training and otherwise
    # divides by zero when that selection is empty. Report failure explicitly.
    if counts.min()<2 or int(.7*len(ids))<2:
        return None,dict(metadata,status='insufficient_training_classes')
    import torch
    from torch.utils.data import TensorDataset,DataLoader
    if not torch.cuda.is_available():raise RuntimeError('Upstream augmentation requires CUDA')
    selected=os.environ.get('PROBE_DEVICE','cuda')
    if not selected.startswith('cuda'):raise RuntimeError('Upstream ScaleDoc requires a CUDA client device')
    selected_device=torch.device(selected)
    torch.cuda.set_device(selected_device.index if selected_device.index is not None else torch.cuda.current_device())
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    models=module('probe_scaledoc_model',source/'model.py')
    sampling=module('probe_scaledoc_sampling',source/'utils/sample.py')
    device=torch.device('cuda')
    td=torch.as_tensor(docs[ids],device=device)
    tq=torch.as_tensor(query,device=device).expand(len(ids),-1)
    data=TensorDataset(torch.stack([td,tq],dim=1),
        torch.as_tensor(labels,dtype=torch.float32,device=device).unsqueeze(1),
        torch.as_tensor(ids,device=device).unsqueeze(1))
    train,valid,aug_count=sampling.aug_split(data,int(.7*len(ids)),3)
    if len(train)==0 or len(valid)==0:return None,dict(metadata,status='empty_training_split')
    train_loader=DataLoader(train,batch_size=batch_size,shuffle=True)
    valid_loader=DataLoader(valid,batch_size=batch_size,shuffle=False)
    encoder=models.Encoder(docs.shape[1]).to(device)
    start=time.perf_counter();history=[]
    for phase,(weights,iterations) in enumerate(zip([(0.,0.,1.),(.2,.8,0.)],epochs)):
        loss=models.ContrastiveLoss(*weights).to(device)
        optimizer=torch.optim.Adam(encoder.parameters(),lr=1e-3,weight_decay=1e-4)
        best=float('inf');best_state=None
        for epoch in range(iterations):
            encoder.train();total=0.
            for inputs,y,_ in train_loader:
                optimizer.zero_grad()
                value,*_=loss(encoder(inputs[:,1],True),encoder(inputs[:,0],True),y)
                if not torch.isfinite(value):return None,dict(metadata,status='nonfinite_training_loss')
                value.backward();optimizer.step();total+=float(value.detach())
            encoder.eval();validation=0.
            with torch.no_grad():
                for inputs,y,_ in valid_loader:
                    value,*_=loss(encoder(inputs[:,1],True),encoder(inputs[:,0],True),y)
                    validation+=float(value)
            validation/=len(valid_loader)
            history.append(dict(phase=phase+1,epoch=epoch,train_loss=total/len(train_loader),validation_loss=validation))
            if np.isfinite(validation) and validation<best:
                best=validation
                if phase==0:best_state={k:v.detach().cpu().clone() for k,v in encoder.state_dict().items()}
        if phase==0:
            if best_state is None:return None,dict(metadata,status='no_valid_checkpoint')
            encoder.load_state_dict(best_state)
    encoder.eval();scores=[]
    with torch.no_grad():
        qe=encoder(torch.as_tensor(query,device=device).unsqueeze(0),False)
        for start_idx in range(0,len(docs),batch_size):
            de=encoder(torch.as_tensor(docs[start_idx:start_idx+batch_size],device=device),False)
            scores.append(torch.nn.functional.cosine_similarity(de,qe,dim=1).cpu().numpy())
    result=np.concatenate(scores)
    return result,dict(metadata,status='trained',augmented_count=int(aug_count),
                       seconds=time.perf_counter()-start,history=history)


def calibration_indices(scores, bins, fraction, *, count=None):
    """Local largest-remainder calibration, including the final histogram edge.

    The local vendored implementation already replaces upstream per-bin floor
    allocation with a fixed total. Keep that adaptation explicit; fix its lost
    maximum-score rows and preserve sampling without replacement within bins.
    """
    if not 0 <= fraction <= 1:
        raise ValueError('Calibration fraction must be between zero and one')
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if not n or fraction == 0:
        return np.empty(0, dtype=int)
    total = min(n, max(1, int(fraction*n))) if count is None else int(count)
    if not 0 <= total <= n:raise ValueError('Invalid exact calibration count')
    bucket = np.searchsorted(bins, scores, side='right') - 1
    bucket = np.clip(bucket, 0, len(bins)-2)
    groups = [np.flatnonzero(bucket == i) for i in range(len(bins)-1)]
    exact = np.array([len(g)*total/n for g in groups])
    take = np.floor(exact).astype(int)
    for i in np.argsort(-(exact-take), kind='stable')[:total-int(take.sum())]:
        take[i] += 1
    return np.concatenate([np.random.choice(g, k, replace=False)
                           for g,k in zip(groups,take) if k])


def cascade(scores,pool,label_ids,*,seed=0,target=.9,calibration_fraction=.05,calibration_count=None):
    """label_ids(original IDs, stage) must raise on unresolved oracle labels.

    No population labels are accepted. Returns only decisions for the remaining
    pool; caller retains and charges the training labels independently.
    """
    src=source_path()
    sys.path.insert(0,str(src))
    spec=importlib.util.spec_from_file_location('probe_scaledoc_cascade',src/'cascade.py')
    C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
    np.random.seed(seed)
    pool=np.asarray(pool,dtype=int);scores=np.asarray(scores,dtype=float)
    if len(pool)!=len(scores) or not np.isfinite(scores).all():raise ValueError('Pool and scores must align')
    if len(pool)==0:return np.empty(0,bool),dict(calibration_ids=[],fallback_ids=[])
    hist,bins=np.histogram(scores,bins=64)
    sampled=calibration_indices(scores,bins,calibration_fraction,count=calibration_count)
    from oracle_budget import binary_labels_or_raise
    labels=binary_labels_or_raise(label_ids(pool[sampled],'calibration'))
    if len(labels)!=len(sampled):raise ValueError('Calibration response length mismatch')
    pos,_=np.histogram(scores[sampled[labels==1]],bins=bins)
    neg,_=np.histogram(scores[sampled[labels==0]],bins=bins)
    pos=pos+np.random.choice([0,.1,.2],size=pos.shape,p=[.6,.3,.1])
    neg=neg+np.random.choice([0,.1,.2],size=neg.shape,p=[.6,.3,.1])
    _,neg=C.smooth_distr(bins,neg,window_size=5)
    _,pos=C.smooth_distr(bins,pos,window_size=5)
    centers=(bins[:-1]+bins[1:])/2;steps=bins[::2]
    lo,hi,_=C.select_sim_filterB(steps,centers,pos,neg,bins[0],bins[-1],target)
    # Same bin widening as upstream apply_bounds; avoid its GT-only metrics API.
    lo=bins[max(0,np.where(steps==lo)[0][0]*2-1)]
    hi=bins[min(len(bins)-1,np.where(steps==hi)[0][0]*2+1)]
    decision=np.where(scores<lo,0,np.where(scores>=hi,1,-1))
    ambiguous=np.flatnonzero(decision==-1)
    resolved=binary_labels_or_raise(label_ids(pool[ambiguous],'fallback'))
    if len(resolved)!=len(ambiguous):raise ValueError('Fallback response length mismatch')
    decision[ambiguous]=resolved
    # Labels already purchased in calibration override approximate predictions.
    # This adapter behavior is explicit and benefits the baseline.
    decision[sampled]=labels
    return decision==1,dict(calibration_ids=pool[sampled].tolist(),
        fallback_ids=pool[ambiguous].tolist(),lower=float(lo),upper=float(hi),
        target=target,branch='upstream F1',verified_calibration_override=True,
        calibration_design='fixed-total largest remainder over 64 bins; final edge included')
