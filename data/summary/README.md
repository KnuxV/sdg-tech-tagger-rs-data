# Summary tables

Counts and shares computed on the 300,001 works of this record. They are
aggregates of the files in `data/annotations/` and hold no work identifier.
`build/build.py` copies them from the summary script it imports.
Shares are per cent of the works counted.

| File | What it gives |
|---|---|
| `sdg_system_goal.tsv` | works tagged by each SDG system with each goal, and with any goal, in each mode of the port (`corrected`, `compatible`) |
| `sdg_k_systems.tsv` | for each goal and for any goal, how many works are tagged by 0, 1, 2, 3 or 4 of the systems |
| `sdg_by_year.tsv` | works tagged by each system with any goal, per publication year |
| `adt_group_field.tsv` | works tagged by each technology group under the two technology queries (`simplified`, `extended`), in the title, in the abstract, or in either |
| `adt_overlap.tsv` | per group, works tagged by both queries, by `simplified` only, by `extended` only, or by neither |
| `adt_by_year.tsv` | works tagged with any technology under each query, per publication year |
| `llm_positive_counts.tsv` | works the LLM labels positive, per label, under each rule |
| `llm_run_checks.tsv` | checks of the four LLM runs: works read, rows, parse failures, pairs sent to pass 2, answers |
| `llm_sdg_comparison.tsv` | the SDG systems against the LLM, per rule, mode, system and goal |
| `llm_adt_comparison.tsv` | the two technology queries against the LLM, per rule, query and group |

**The LLM reference.** A label is positive for a model when its pass 1
score is at least 0.25 and pass 2 confirms it. The rules are `qwen3`,
`gemma`, `intersection` (both models positive, the reference used in the
text of the package and of this record) and `union` (either model).

**The two comparison figures.**

- `precision_proxy_pct`: among the works a query tags with a label, the
  share the LLM labels positive for the same label.
- `recall_proxy_pct`: among the works the LLM labels positive for a label,
  the share the query tags with it.

Neither is measured against a person: the LLM is the reference, and its
own errors are in both. The `_ci_lo` and `_ci_hi` columns give 95 per cent
Wilson intervals. `unit` says what is counted: a work per label, a (work,
label) pair pooled over labels, or a work with any label. Elsevier, SIRIS
and Auckland have no query for SDG-17, which `system_has_goal_queries`
marks. `below_floor` marks a row with fewer than 10 works on one side.
`UMBRELLA`, `ANY6` and `ANY_WITH_UMBRELLA` are compared with the works the
LLM labels positive for any technology, since UMBRELLA names none.
