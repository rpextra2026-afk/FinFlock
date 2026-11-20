# tessera

this folder contains the anonymized Tessera input dataset, with all 431,817 primary and supplemental records from the supplied package. these are dataset inputs, including the held-out evaluation partitions, rather than model predictions or experiment results.

| component | records |
| --- | ---: |
| financial items | 168,316 |
| geopolitical items | 18,127 |
| social items | 231,111 |
| supplementary geopolitical records | 14,263 |
| company/entity reference table | 3,886 |
| country-day panel | 944,230 |

the supplementary text comprises 14,240 records in `supplement_text/` and 23 records in `geo_supplement_records/public_text.parquet/`. the financial, geopolitical, and social folders each contain item metadata, text, event membership, and entity pins. `MANIFEST.json` lists every table's schema, row count, and every data part's SHA-256 checksum.

## reading the data

tables are stored as partitioned Parquet datasets. a path ending in `.parquet` is a directory containing ordinary Parquet files; pandas and pyarrow read the directory as one table. each file remains below 50 MB for direct GitHub storage.

```python
import pandas as pd

items = pd.read_parquet("dataset/tessera/financial/items.parquet")
```

the standalone loader enforces training and held-out split separation:

```python
import sys
sys.path.insert(0, "dataset/tessera")
from loader.tessera_loader import load_training_inputs, load_heldout

training = load_training_inputs("dataset/tessera", with_text=False)
evaluation = load_heldout(
    "dataset/tessera", split="test", categories=("geopolitical",)
)
```

`with_text=False` avoids loading all article text when only metadata is needed. use `with_text=True` to load training text. the original train, embargo, and test assignments are preserved; held-out data must not enter training.

## anonymization and provenance

author identifiers, credentials, and personal filesystem paths were redacted where found. Parquet schema metadata was removed, and delivered text hashes and lengths were recomputed after redaction. record IDs, memberships, numeric features, timestamps, and split assignments are preserved. public upstream attribution and source references are retained; personal hosting links are not included.

redistribution approval for the full supplied package was confirmed by the uploader. source-specific license, attribution, and rights fields retain their historical values for provenance; this release does not assign a blanket license to upstream material.
