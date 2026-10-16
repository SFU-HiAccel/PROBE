"""PySpark semantic filter and join operators backed by the PROBE Python pipeline."""
from pyspark.sql import DataFrame


def sem_filter_impl(self, prompt, model='', url='http://localhost:8000/v1/chat/completions',
                    embedding_model='', *, proxy_sample=True, random_seed=0,
                    learned_calibration_config=None, **kwargs):
    """Filter rows with PROBE, or evaluate every row when proxy_sample=False."""
    from proxy_model import proxy_filter
    return proxy_filter(self, prompt, model=model, url=url, embedding_model=embedding_model,
                        random_seed=random_seed, learned_calibration_config=learned_calibration_config,
                        no_proxy=not proxy_sample, **kwargs)


def sem_join_impl(self, right, prompt, model='', url='http://localhost:8000/v1/chat/completions',
                  embedding_model='', *, proxy_sample=True, random_seed=0,
                  learned_calibration_config=None, **kwargs):
    """Join candidate pairs with PROBE, or evaluate every pair when proxy_sample=False."""
    from proxy_model import proxy_join
    return proxy_join(self, right, prompt, model=model, url=url, embedding_model=embedding_model,
                      random_seed=random_seed, learned_calibration_config=learned_calibration_config,
                      no_proxy=not proxy_sample, **kwargs)


def register_semantic_operators():
    DataFrame.sem_filter = sem_filter_impl
    DataFrame.sem_join = sem_join_impl


register_semantic_operators()
