# Tessera

Tessera is the point-in-time news-event dataset accompanying FinFlock. it contains **431,817 source records**, **431,312 event IDs**, supported asset/entity links, explicit availability policies, and separate training, embargo, and held-out partitions over 2015–2025. it contributes a common representation for financial disclosures, official geopolitical records, archived news, and investor discussion, with timestamp provenance and leakage controls carried into model inputs.

## dataset overview

| Part | Sources | Items | Events | Train / test items | Avg. text | Availability-time basis | Asset links / exposure | Historical release assessment (3 Nov 2026) |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| **Company filings** | SEC 8-K; foreign 6-K/20-F (14 event families) | 168,316 | 167,967 | 137,657 / 30,366 | 4.5 KB; 6,000-character excerpt cap | **100% exact** SEC acceptance timestamps, to the second | 100% supported company links | IDs and times cleared; text unresolved at assessment |
| **Official geopolitical** | Federal Register, OFAC, GDELT 1.0, Wikinews (9 families) | 18,127 | 17,971 | 12,675 / 5,404 | 0.4 KB | **74.8% exact**; remaining items use conservative date-only timing | 53% country; 20% commodity; no company labels | Full metadata and text cleared |
| **Geopolitical news supplement** | 11 news archives, including NYT and Yahoo Finance | 14,263 | 14,263 | 9,856 / 4,402 | 1.28 KB; 46.4% below 200 characters | Date-only policy; edit history unavailable | 33.3% company exposure, unaudited; 48.1% strictly country-only exposure | Most publisher text restricted; 23 Wikinews copies cleared |
| **Social** | Reddit: r/investing, r/stocks, r/StockMarket, r/wallstreetbets | 231,111 | 231,111 | 182,332 / 47,567 | 0.6 KB | Date-only policy; source clock claims are not promoted to exact | 18.5% supported company links; 68.4% have no recorded pin | Permission unresolved at assessment |
| **Total** | **21 feeds; 25 event families** | **431,817** | **431,312** | **342,520 / 87,739** | **923.6 MB total UTF-8 text** | **42.1% exact** | **3,886 company CIKs** | **18,150 full-text records (4.2%); 186,466 metadata records (43.2%) cleared at assessment** |

**Table 1. Tessera news-event dataset, 2015-01-01 to 2025-12-31 (UTC).** train contains items assigned to availability sessions on or before 2023-12-31; the held-out test covers 2024-01-09 to 2025-12-31 after a five-session embargo. the 1,558 embargo items are included in the item totals and omitted from the train/test column. “exact” identifies an official SEC acceptance timestamp or Federal Register public-inspection timestamp, rather than an unverified clock claim. date-only items become available at the start of the next UTC day; GDELT uses a three-day delay. the 431,312 events comprise 417,049 core events and 14,263 distinct supplemental event IDs. text sizes describe the frozen source text before anonymization, in decimal bytes: 905.3 MB for the three core parts plus 18.3 MB for the supplement. the country-day panel adds 944,230 structured rows across 47 countries and five GDELT event families.

**Release-status context.** the last column preserves the historical source assessment, not a claim that every included source shares one license. full-package redistribution approval was subsequently confirmed by the uploader, and the current repository includes all four components. historical rights fields remain as provenance, and source attribution is retained. the full-text and metadata figures overlap; they are not additive.

these are input records and construction statistics. saved reader predictions, persona responses, fitted models, backtest results, and other experiment outputs are not included.

## source composition

| Component | Source records |
| --- | --- |
| Company filings | SEC 8-K: 167,261; foreign 6-K/20-F: 1,055 |
| Official geopolitical | Federal Register: 13,568; OFAC: 765; GDELT: 3,735; Wikinews: 59 |
| News supplement | NYT articles: 1,857; NYT headlines: 463; Yahoo Finance articles: 4,345; additional Yahoo Finance archive: 827; multi-outlet headlines: 6,285; HuffPost: 66; American news archive: 325; FinSen: 35; CNBC: 3; market headlines: 34; Wikinews archive: 23 |
| Social | r/investing: 64,293; r/stocks: 63,682; r/StockMarket: 49,609; r/wallstreetbets: 53,527 |

the 21 feeds are represented by two filing sources, four official sources, eleven supplemental archives, and four social communities. the 25 event families comprise fourteen filing families, nine official geopolitical families, one supplemental family, and one social family. the source-pool stage contained 4,381,993 items and excluded 3,322,788 rights-limited archive items from that stage; subsequent social and supplemental collection stages have their own screening and selection counts, so this earlier pool is not the final dataset's denominator.

## records, events, and pins

- **Item:** one source record, such as a filing, notice, social post, or supplemental article. items retain their source and event IDs.
- **Event:** a cluster of related core items, formed using entity sets and normalized titles within 48 hours. social posts are one item per event; the supplement has 14,263 distinct event IDs. event membership is explicitly stored.
- **Pin:** an item-to-company, country, commodity, or other named-entity relation with a source field and confidence. company identities use CIKs, and the reference table contains 3,886 companies.
- **Text:** stored separately from metadata and joined by `item_id` or `record_id`. SEC text is an excerpt capped at 6,000 characters, with 51.6% reaching the cap; it remains filing prose rather than being relabeled as news.

the supplementary text has 14,240 records in `supplement_text/` and 23 in `geo_supplement_records/public_text.parquet/`. `common/gdelt_country_day.parquet/` contains the 944,230-row structured panel. the complete release has **18 logical tables in 93 Parquet parts**. `MANIFEST.json` records schemas, row counts, file sizes, and SHA-256 checksums.

## availability timestamps

`published_utc` stores the source stamp; `available_utc` stores the policy instant used for point-in-time access; `policy_session` assigns the first XNYS session whose close follows that instant. these fields distinguish a source's clock claim from the timestamp the dataset can substantiate.

SEC acceptance times have second precision. 13,567 Federal Register notices have official public-inspection times, recorded to the minute; one further notice falls back to date-only timing. the other official sources, social posts, and news archives use conservative date-only availability. GDELT receives a three-day delay, while the other date-only records use the next UTC day. a raw publication clock on a news or social record is retained as a claim and is not promoted to exact.

the supplement has no complete edit history, and its Wikinews archive copies lack revision IDs. conservative availability therefore does not establish that an archived text is exactly the version available at its claimed publication time.

## temporal partitions and framework evaluation

| Partition | Items | Availability-session policy |
| --- | ---: | --- |
| Training (`train`; source split `dev`) | 342,520 | Up to 2023-12-31 |
| Embargo (`heldout_embargo`) | 1,558 | Five sessions, 2024-01-02 through 2024-01-08 |
| Held-out test (`heldout_test`) | 87,739 | 2024-01-09 through 2025-12-31 |
| **Total** | **431,817** | |

the standalone loader rejects held-out or unverifiable rows passed to a training guard and loads held-out partitions only through the evaluation interface. FinFlock's framework experiments use 2015–2021 for fitting and 2022–2023 for holdout analysis inside the package training period, followed by the separate sealed 2024–2025 interval. the package's embargo and held-out membership still apply within that sealed interval.

## asset links and exposure

company-label coverage uses supported high- or medium-confidence pins and excludes learned-headline and current-ticker-only fields. country and commodity links do not create company labels, and an unlinked social post remains unlabelled.

- All 168,316 filing items have supported company links.
- Official geopolitical items have 9,635 country-linked records (53.2%) and 3,542 commodity-linked records (19.5%); these sources do not receive company labels.
- Social records have 42,751 supported company labels (18.5%). 158,138 posts (68.4%) have no recorded pin; 172,549 (74.7%) have no supported pin. the latter also includes records whose only pins fail the support rule.
- The supplement has 4,747 automatically company-exposed records (33.3%), based on alias matching rather than a recorded human accuracy audit. 6,860 records (48.1%) are strictly country-only: country exposure is present and company, commodity, and sector-term exposure are absent. the separately reported 57.0% figure is the P3 priority-tier share, not this strict country-only measure.

these are coverage measures, not estimates of linking accuracy. no filled human-audit judgments were recorded in the frozen package, so the release does not claim a passed human accuracy audit.

## construction checks

the frozen package validation reported:

- unique IDs, valid event memberships and pin references, and matching delivered-text hashes;
- zero events, URLs, or source pointers spanning the training/held-out boundary;
- zero cross-boundary SEC near-duplicate pairs under the same-CIK, four-gram Jaccard ≥ 0.8, 45-day rule;
- removal of 147 training SEC filings through the near-duplicate and exact-duplicate corrections, removal of 621 self-reported edited social posts, and corrected availability for 227 supplement records;
- 147 flagged official-document template groups involving 811 notices across partitions, retained as distinct documents with different source identifiers; the loader can exclude their training members with `exclude_template_crossing=True`;
- enforcement of the training split guard and no invented company labels for unlinked or country/commodity-only records.

these are the source package's construction checks, not a new rerun of every collection-stage audit. the current distribution additionally carries per-file checksums and preserves IDs, timestamps, numerical features, and partition assignments through anonymization. text volume and average-length figures above were checked against the frozen source tables; the strict supplement exposure definition is stated explicitly to resolve ambiguity in the earlier report.

## coverage and interpretation

the supplementary archive has uneven temporal coverage: narrative geopolitical coverage in 2024 is sparse, and the 2025 narrative test material is dominated by one Yahoo Finance feed. 6,611 supplement records (46.4%) have fewer than 200 characters. r/wallstreetbets has no records in 2022–2023 in the supplied dumps. alias-based exposure links and uncertain archive versions should be accounted for when interpreting results.

Tessera does not itself contain market prices, returns, model predictions, or evidence that a language model was not pretrained on its held-out text. evaluation of FinFlock additionally requires the separate market and investor inputs documented in the main README.

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
