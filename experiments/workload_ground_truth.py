"""Evaluation-only preparation of human annotations; never proxy training labels."""
import hashlib
import json
from pathlib import Path


def cars_q7_image_ground_truth(data_root, scale, output):
    """Restrict SemBench's dented-image annotation to the scale's image universe.

    The full Q7 car-ID union includes text/audio predicates and cannot score
    the 30,130-image case study. Fail on missing/ambiguous annotations.
    """
    import pandas as pd
    root = Path(data_root) / 'SemBench/files/cars/data'
    images = root / f'sf_{scale}/image_car_data_{scale}.csv'
    annotations = root / 'full_data/image_data_full.csv'
    keys = ['image_id', 'car_id']
    universe = pd.read_csv(images, dtype=str, usecols=keys)
    labels = pd.read_csv(annotations, dtype=str, usecols=keys+['damage_status'])
    joined = universe.merge(labels, on=keys, how='left', validate='many_to_one')
    if joined['damage_status'].isna().any():
        raise ValueError('Missing Cars Q7 image annotations in the candidate universe')
    positives = joined.loc[joined['damage_status'].str.contains('dented', regex=False), ['car_id']]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'cars_q7_image_gt_{scale}.csv'
    positives.drop_duplicates().sort_values('car_id').to_csv(path, index=False)
    provenance = dict(query='cars_q7', variant='image_only_dented', candidates=len(universe),
                      positive_rows=len(positives), evaluation_only=True,
                      source_sha256={str(p.relative_to(Path(data_root))):hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (images, annotations)})
    path.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return str(path)


def animals_q1_ground_truth(data_root, scale, output):
    """Human row labels for the paper's zebra filter, rather than count accuracy."""
    import pandas as pd
    source = Path(data_root)/f'SemBench/files/animals/data/sf_{scale}/image_data.csv'
    frame = pd.read_csv(source)
    if frame['Species'].isna().any():
        raise ValueError('Animals Q1 requires a species annotation for every image row')
    positive = frame['Species'].str.contains('ZEBRA',regex=False).to_numpy()
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/f'animals_q1_rows_sf{scale}.csv'
    pd.DataFrame({'row_id':frame.index[positive]}).to_csv(path,index=False)
    path.with_suffix('.provenance.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        candidates=len(frame),positive_rows=int(positive.sum()),metric='binary_row_f1'),indent=2)+'\n')
    return str(path)
