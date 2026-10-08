"""Baseline policies over the same messages, candidate IDs and expert transport.

CSV follows its published equal-vote recursion. LOTUS uses the vendored upstream
threshold learner; ScaleDoc uses its explicitly configured upstream source tree.
No baseline sees benchmark ground truth.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np


BASELINE_MODELS = {
    'text': dict(model='intfloat/e5-base-v2', revision='f52bf8ec8c7124536f0efb74aca902b2995e5bcd'),
    'multimodal': dict(model='openai/clip-vit-base-patch32', revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'),
}


def encode_contents(left, right, predicate, job, device):
    """Frozen, label-free text/image features; never replace an image with text."""
    from huggingface_hub import snapshot_download
    import torch
    from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
    from PIL import Image
    multimodal = any(r.get('image') for r in (left or [])+right)
    chosen = dict(BASELINE_MODELS['multimodal' if multimodal else 'text'])
    chosen.update(job.get('baseline_representation', {}).get('multimodal' if multimodal else 'text', {}))
    snapshot = snapshot_download(chosen['model'], revision=chosen['revision'])
    if multimodal:
        model = CLIPModel.from_pretrained(snapshot).to(device).eval()
        processor = CLIPProcessor.from_pretrained(snapshot)
    else:
        model = AutoModel.from_pretrained(snapshot).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(snapshot)
    def encode(items, *, images=False):
        with torch.inference_mode():
            if multimodal:
                inputs = processor(images=items, return_tensors='pt') if images else processor(
                    text=items, padding=True, truncation=True, max_length=77, return_tensors='pt')
                inputs = {k:v.to(device) for k,v in inputs.items()}
                result = model.get_image_features(**inputs) if images else model.get_text_features(**inputs)
                vectors = result.pooler_output if hasattr(result, 'pooler_output') else result
            else:
                inputs = tokenizer(items, padding=True, truncation=True, max_length=512,
                                   return_tensors='pt').to(device)
                hidden = model(**inputs).last_hidden_state
                weights = inputs['attention_mask'].unsqueeze(-1)
                vectors = (hidden*weights).sum(1)/weights.sum(1).clamp_min(1)
            return torch.nn.functional.normalize(vectors.float(), dim=1).cpu().numpy()
    def side(rows):
        entries = list(dict.fromkeys((r.get('text') or '', r.get('image')) for r in rows))
        lookup = {r:i for i,r in enumerate(entries)}
        vectors = []
        for start in range(0,len(entries),32):
            chunk = entries[start:start+32]
            text = [t if multimodal else 'passage: '+t for t,_ in chunk]
            enc = encode(text)
            image_positions = [i for i,(_,p) in enumerate(chunk) if p]
            images = []
            try:
                for i in image_positions:
                    with Image.open(chunk[i][1]) as image:
                        images.append(image.convert('RGB'))
                if images:
                    ie = encode(images, images=True)
                    for i,v in zip(image_positions,ie):
                        enc[i] = v if not chunk[i][0] else (enc[i]+v)/2
            finally:
                for image in images:image.close()
            enc /= np.maximum(np.linalg.norm(enc,axis=1,keepdims=True),1e-12)
            vectors.append(enc)
        unique = np.concatenate(vectors)
        return unique[[lookup[(r.get('text') or '',r.get('image'))] for r in rows]]
    b = side(right)
    features = b if left is None else (side(left)+b)/2
    features /= np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
    query = encode([predicate if multimodal else 'query: '+predicate])[0]
    del model
    return features, query, dict(chosen, fusion='mean of normalized sides; preserve text and images',
                                 snapshot_revision=Path(snapshot).name)


def csv_vote(features, oracle, *, fraction, seed, max_depth=3):
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(features),bool)
    trace = []
    def recurse(pool, depth):
        if not len(pool):return
        clusters = min(4 if depth == 0 else 2, len(pool))
        assignments = KMeans(n_clusters=clusters,random_state=seed,n_init=10).fit_predict(features[pool])
        for cluster in range(clusters):
            members = pool[assignments == cluster]
            if not len(members):continue
            k = min(len(members),max(1,int(fraction*len(members))))
            sampled = rng.choice(members,k,replace=False)
            labels = oracle.label_ids(sampled,stage='L')
            mask[sampled] = labels.astype(bool)
            remaining = np.setdiff1d(members,sampled)
            rate = float(labels.mean())
            trace.append(dict(depth=depth,cluster=cluster,size=len(members),sample_ids=sampled.tolist(),positive_fraction=rate))
            if rate >= .9:mask[remaining] = True
            elif rate <= .1:mask[remaining] = False
            elif depth+1 >= max_depth:mask[remaining] = oracle.label_ids(remaining,stage='fallback').astype(bool)
            else:recurse(remaining,depth+1)
    recurse(np.arange(len(features)),0)
    return mask, dict(clusters=[4,2],consensus=.9,sample_fraction=fraction,
                      min_samples=1,max_depth=max_depth,trace=trace)


def lotus_cascade(texts, oracle, proxy, *, target, seed, output):
    from openai import OpenAI
    from third_party.lotus.cascade_utils import (CascadeArgs, calibrate_llm_logprobs,
                                                importance_sampling, learn_cascade_thresholds)
    # Use the identical query-specific expert prompt for the helper model.
    request = oracle.oracle.request_factory()
    system_prompt = request.system_prompt
    client = OpenAI(base_url=proxy['oracle_url'].removesuffix('/chat/completions'),api_key='EMPTY',max_retries=0)
    probabilities = []
    from concurrent.futures import ThreadPoolExecutor
    concurrency = 128
    journal = output/'proxy_oracle.jsonl'
    def request(i):
        return client.chat.completions.create(model=proxy['model'],
            messages=[dict(role='system',content=system_prompt),dict(role='user',content=texts[i])],
            temperature=0,max_tokens=16,top_p=1,presence_penalty=0,frequency_penalty=0,
            logprobs=True,top_logprobs=20,
            extra_body={'top_k':-1,'repetition_penalty':1.,
                        'chat_template_kwargs':{'enable_thinking':False}})
    with ThreadPoolExecutor(max_workers=concurrency) as workers, journal.open('a') as stream:
        # Bound the queued responses as well as active HTTP calls on large joins.
        for start in range(0,len(texts),concurrency):
            ids=range(start,min(len(texts),start+concurrency))
            for i,response in zip(ids,workers.map(request,ids)):
                tokens = response.choices[0].logprobs
                if tokens is None:
                    stream.write(json.dumps(dict(id=i,response=response.model_dump(),
                                                 error='missing token log-probabilities'))+'\n')
                    raise RuntimeError('LOTUS proxy endpoint must return token log-probabilities')
                probability = None
                for token in tokens.content:
                    choices={t.token:float(np.exp(t.logprob)) for t in token.top_logprobs}
                    # Exact True/False convention of LOTUS's upstream helper formatter.
                    if 'True' in choices and 'False' in choices:
                        probability=choices['True']/(choices['True']+choices['False']);break
                if probability is None:
                    probability=float(any(t.token=='True' for t in tokens.content))
                if not np.isfinite(probability):raise RuntimeError('Nonfinite LOTUS proxy probability')
                probabilities.append(probability)
                stream.write(json.dumps(dict(id=i,response=response.model_dump(),true_probability=probability))+'\n')
    args=CascadeArgs(recall_target=target,precision_target=target,failure_probability=.1,
                     sampling_percentage=.1,cascade_IS_random_seed=seed)
    scores=calibrate_llm_logprobs(probabilities,args)
    sample,correction=importance_sampling(scores,args)
    if len(sample)==0:
        return oracle.label_ids(range(len(texts)),stage='fallback').astype(bool),dict(status='empty_calibration_full_oracle')
    labels=oracle.label_ids(sample,stage='A').astype(bool)
    (high,low),_=learn_cascade_thresholds([scores[i] for i in sample],labels.tolist(),correction[sample],args)
    scores=np.asarray(scores)
    mask=scores>=high
    routed=np.flatnonzero((scores>low)&(scores<high))
    mask[routed]=oracle.label_ids(routed,stage='D').astype(bool)
    for i,y in oracle.ledger.cache.items():mask[i]=bool(y)
    return mask,dict(lower=float(low),upper=float(high),sample_ids=sample.tolist(),
                     routed_ids=routed.tolist(),proxy_requests=len(probabilities),
                     upstream_version='lotus-ai 1.1.3 vendored cascade helper',
                     proxy_concurrency=concurrency,proxy=proxy)


def execute_baseline(job, features, query_embedding, texts, oracle, output):
    system=job['system'];point=job['point'];n=len(texts);seed=job['seed']
    before=time.perf_counter();calls=oracle.seconds
    if system=='CSV':
        mask,metadata=csv_vote(features,oracle,fraction=point.get('budget_pct',10)/100,seed=seed)
    elif system=='AIQA':
        from cluster_proxy import sample_uqe_active
        from sklearn.linear_model import LogisticRegression
        budget=int(n*point.get('budget_pct',10)/100)
        ids,y,_,trace=sample_uqe_active(features,None,texts,oracle,budget,
            batch_size=16,exploration_sigma_0=1.,exploration_decay=.9,random_seed=seed)
        if len(np.unique(y))<2:
            # A pure sampling-budget baseline has no subsequent cascade.
            mask=np.full(n,bool(y[0]) if len(y) else False)
        else:
            lr=LogisticRegression(max_iter=1000,random_state=seed)
            lr.fit(features[ids],y);mask=lr.predict(features).astype(bool);mask[ids]=y.astype(bool)
        metadata=dict(sampler=trace,training_ids=ids.tolist(),training_budget=budget,
                      single_class_policy='constant prediction; no additional expert calls')
    elif system=='LOTUS':
        mask,metadata=lotus_cascade(texts,oracle,job['proxy'],target=point.get('target',.9),seed=seed,output=output)
    elif system=='ScaleDoc':
        from experiments.scaledoc_adapter import train_scores,cascade
        ids=np.random.default_rng(seed).choice(n,int(.05*n),replace=False)
        labels=oracle.label_ids(ids,stage='training')
        scores,training=train_scores(features,query_embedding,ids,labels,seed=seed)
        if scores is None:
            mask=oracle.label_ids(range(n),stage='fallback').astype(bool)
            metadata=dict(training=training,status='insufficient_training_full_oracle')
        else:
            pool=np.setdiff1d(np.arange(n),ids)
            # 5% of full N for calibration, excluding already purchased training rows.
            decisions,routing=cascade(scores[pool],pool,lambda ids,stage:oracle.label_ids(ids,stage=stage),
                seed=seed,target=point.get('target',.9),calibration_fraction=.05,
                calibration_count=min(len(pool),int(.05*n)))
            mask=np.zeros(n,bool);mask[pool]=decisions;mask[ids]=labels.astype(bool)
            metadata=dict(training=training,routing=routing)
    else:raise ValueError(system)
    metadata['stages']=dict(sampling=max(0.,time.perf_counter()-before-(oracle.seconds-calls)))
    return mask,metadata
