from .sample import SampleDataset
from .flood import FloodDataset
from .tide import TideDataset

__datasets = {
    "flood": FloodDataset,
    "tide": TideDataset,
}

def init_dataset(name, **kwargs):
    """Initializes a dataset."""
    avai_datasets = list(__datasets.keys())
    if name not in avai_datasets:
        raise ValueError('Invalid dataset name. Received "{}", '
                         'but expected to be one of {}'.format(
                             name, avai_datasets))
    return __datasets[name](**kwargs)