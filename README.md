# sdg-tech-tagger-rs data: 300,001 OpenAlex works tagged by every method

A random sample of 300,001 OpenAlex works, tagged with sustainable
development goals by the four query systems of text2sdg (Aurora, Elsevier,
SIRIS, Auckland), with digital technologies by two query sets, and by two
large language models. It holds two sets:

1. **R against Rust.** Every firing of text2sdg 1.1.2, the R package, and of
   its port `sdg-tech-tagger-rs`, in both modes, with the words each matched.
   It shows that the port returns what R returns on texts it was never
   developed on.
2. **The works fully annotated.** For each work: the technology queries, the
   four SDG systems in both modes, and the LLM labels.

Identifiers and tags only: no title, abstract or DOI is included. The
`features` columns hold the words a query matched, taken verbatim from the
title and abstract, a few words per firing. The texts can be rebuilt from
OpenAlex with the work identifiers (see below).

## The sample

- **Frame:** every OpenAlex work published 2019 to 2024 that is an article
  or a review, in English, not in XPAC, not retracted, and with an abstract:
  27,328,980 works, in the OpenAlex snapshot of 2026-02-03.
- **Draw:** 300,001 works, proportional to publication year by field, each
  work drawn by a fixed pseudo-random key.
- **Mix:** 285,540 articles and 14,461 reviews; 46,380 to 52,744 works per
  year.
- **Text the tools read:** the title, then `". "`, then the abstract, with
  publishers' rights notices stripped from the abstract. 299,993 works have a
  non-empty abstract after stripping.

## The tools

- **text2sdg 1.1.2** (Wulff and Meier; Meier, Mata and Wulff 2024), run in R
  with its systems Aurora, Elsevier, SIRIS and Auckland.
- **sdg-tech-tagger-rs 0.1.0**, the port: mode `compatible` returns what
  text2sdg returns; mode `corrected`, its default, applies corrections that
  restore the published Aurora queries.
- **Technology queries:** `simplified` (six groups) and `extended` (the
  same six groups with more terms each, and `UMBRELLA`). Both ship with the
  package, which lists every term in `docs/technology-queries.md`.
- **LLM:** two models, `qwen3` (Qwen 3.6 35B A3B) and `gemma` (Gemma 4 31B),
  under the frozen prompts of protocol v1, which ship with the package. Pass
  1 scores every label of every work; pass 2 asks again for every (work,
  label) whose pass 1 score is at least 0.25. A label is **positive** for a
  model when its pass 1 score is at least 0.25 and pass 2 answers `Y` (SDG)
  or one of `E`, `A`, `T` and `M` (technology: every answer but `N`, the
  no, as the prompts of protocol v1 define them; a mention counts). The **consensus** is positive when both models are positive.

## Set 1: R against Rust

<!-- agreement:start -->
The agreement table of set 1 is written here by `build/build.py` once both R runs are complete.
<!-- agreement:end -->

On a first random subset of 2,000 of these works, R and the port gave the
same 2,746 firings in compatible mode and the same 2,729 in corrected mode,
with the same matched words, and none on one side only.

## Files

All tables are Parquet. `work_id` is the OpenAlex identifier without its
URL prefix (`W...`); `https://openalex.org/` + `work_id` is the work.

| File | One row per | Columns |
|---|---|---|
| `data/works.parquet` | work | `work_id`, `publication_year`, `type` (article, review), `field`, `subfield` (OpenAlex topics), `has_abstract`, `text_chars`, `text_sha256` |
| `data/annotations/sdg.parquet` | firing of the port | `work_id`, `mode` (corrected, compatible), `system`, `sdg` (SDG-01 to SDG-17), `query_id` (the query's row in text2sdg's table), `features` (the matched words, comma separated) |
| `data/annotations/technology.parquet` | (work, query, group) that fires | `work_id`, `query` (simplified, extended), `group` (AI, IOT, CI, BL, ROB, AM, UMBRELLA), `title`, `abstract` (Booleans: where it fires) |
| `data/annotations/llm.parquet` | (work, exercise, label) | `work_id`, `exercise` (sdg, adt), `label` (SDG-01 to SDG-17; AI, IOT, COMPUTING_INFRASTRUCTURE, BLOCKCHAIN, ROBOTICS, ADDITIVE_MANUFACTURING), then for each model `<model>_pass1_score`, `<model>_pass2_answer` (null when pass 2 was not asked; `OTHER` when the answer was no letter), `<model>_positive`; and `consensus_positive` |
| `data/r_vs_port/firings.parquet` | (work, mode, system, sdg, query_id) fired by R or the port | `work_id`, `mode`, `system`, `sdg`, `query_id`, `features_r`, `features_port`, `in_r`, `in_port`, `features_equal` |
| `data/r_vs_port/agreement.tsv` | (mode, system) | `firings`, `both`, `port_only`, `r_only`, `features_equal` |
| `data/summary/*.tsv` | see `data/summary/README.md` | aggregate counts and the comparison with the LLM |
| `data/SHA256SUMS` | file | checksums of everything under `data/` |

A work absent from `sdg.parquet` or `technology.parquet` is a work on which
nothing fired. Technology rows exist only where the group fires in the title
or the abstract. The LLM table holds every work and label, positive or not.

## Rebuilding the texts

The texts are not included: abstracts belong to their publishers. To rebuild
one, ask OpenAlex for the work and its abstract, which it serves as an
inverted index, and put each word back at its positions:

```python
import requests

def rebuild(index: dict[str, list[int]]) -> str:
    """Return the abstract from OpenAlex's inverted index."""
    words = {pos: word for word, positions in index.items() for pos in positions}
    return " ".join(words[i] for i in sorted(words))

work = requests.get(
    "https://api.openalex.org/works/W2741809807",
    params={"select": "id,title,abstract_inverted_index"},
).json()
text = work["title"] + ". " + rebuild(work["abstract_inverted_index"] or {})
```

The tools read the abstract as stored in the copy of the snapshot used, with
rights notices stripped, so a rebuilt text can differ in spacing or
punctuation; `text_sha256` in `works.parquet` tells whether it is the same
text, byte for byte.

## How the files are made

`build/build.py` reads the sample, the stored runs and the R outputs, runs
the port, and writes `data/`. It checks that the stored corrected-mode
firings equal a fresh run of the port on 5,000 works, and that the LLM
positives equal the counts in `data/summary/llm_positive_counts.tsv`. It
writes set 1 only when every chunk of both R runs is there. It was run in
the environment where the sample was drawn, and is given for transparency:
its inputs, which hold the abstracts, are not part of the record.

## Licence and citation

The tables are under CC BY 4.0 (`LICENSE`). OpenAlex metadata is CC0. Cite
this record, the package `sdg-tech-tagger-rs`, text2sdg, and OpenAlex; see
`CITATION.cff`.

## Before publishing

To do before this record goes public; remove this section then.

- [ ] Add set 1: run `build/build.py` with `--r-dir` once both R runs are
      complete, and check the agreement block above.
- [ ] Settle the layout of the files (proposed).
- [ ] Add the DOI of the `sdg-tech-tagger-rs` Zenodo record to
      `.zenodo.json` (`related_identifiers`, relation `isCompiledBy`,
      resource type `software`) and to `CITATION.cff`.
- [ ] Decide the release date.
- [ ] Tag `v0.1.0`.
- [ ] Regenerate `data/SHA256SUMS` (the build script does it).
