"""Build the files of the data record from the recall sample and the tools' runs.

Consumes, read-only:
- the prepared recall sample of open_alex_adt_gate (300,001 OpenAlex works,
  with the `text` the tools read and the flags of the two technology queries);
- the port's SDG firings in corrected mode stored there (`recall_sdg_hits`);
- the eight LLM runs on the sample (SDG and technology, pass 1 and 2, qwen3
  and gemma);
- for set 1, the R runs of text2sdg 1.1.2 in both modes, written by
  `r/detect_text2sdg.R` and `r/detect_text2sdg_corrected.R` of
  text2sdg_rust_port, sharded as `<r-dir>/{compat,corr}/shard_<i>/chunks/`.

Runs the port itself in compatible mode on the sample's text, and in
corrected mode on a subset to check the stored firings.

Produces, under `data/`: `works.parquet`, `annotations/{sdg,technology,llm}.parquet`,
`r_vs_port/{firings.parquet,agreement.tsv}` (set 1, only when every R chunk
is there), and `SHA256SUMS`; and fills the agreement block of `README.md`.
No title, abstract or DOI is written.

The LLM semantics are those of text2sdg_rust_port's
`probes/recall_summary_probe/summarise.py`, imported, not re-derived: a label
is positive for a model when its pass 1 score is at or above 0.25 and pass 2
answers a positive letter; the consensus is positive under both models.

Usage, from the root of this repository:
    uv run --project /home/knuxv/Codebase/text2sdg_rust_port python build/build.py \
        --r-dir <directory of the R runs>
    ... --skip-r    to write set 2 alone
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import polars as pl
from text2sdg_rust_port import detect

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PORT_REPO = Path("/home/knuxv/Codebase/text2sdg_rust_port")
PROBE = PORT_REPO / "probes/recall_summary_probe/summarise.py"
SHARDS = 5
CHUNK = 2000
CHECK_SUBSET = 5000
OPENALEX = "https://openalex.org/"
MODES = {"compat": "compatible", "corr": "corrected"}
# Technology groups: column code in the sample -> code in this record.
TECH = {
    "ai": "AI",
    "iot": "IOT",
    "ci": "CI",
    "bl": "BL",
    "rob": "ROB",
    "am": "AM",
    "umbrella": "UMBRELLA",
}
README_START = "<!-- agreement:start -->"
README_END = "<!-- agreement:end -->"


def load_probe():
    """Import the summary probe, the one place the LLM semantics are written."""
    spec = importlib.util.spec_from_file_location("summarise", PROBE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["summarise"] = module
    spec.loader.exec_module(module)
    return module


def short_id(col: str) -> pl.Expr:
    """Turn a full OpenAlex URL into its W identifier, named work_id."""
    return pl.col(col).str.strip_prefix(OPENALEX).alias("work_id")


def load_sample(interim: Path) -> pl.DataFrame:
    """Return the prepared sample: ids, metadata, text and technology flags."""
    df = pl.read_parquet(interim / "recall_sample_prepared.parquet")
    assert df.height == df["id"].n_unique() == 300_001
    assert df["id"].str.starts_with(OPENALEX + "W").all()
    return df


def works_table(sample: pl.DataFrame) -> pl.DataFrame:
    """One row per work: metadata and the checksum of the text the tools read."""
    sha = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in sample["text"]]
    return sample.select(
        short_id("id"),
        "publication_year",
        "type",
        "field",
        "subfield",
        "has_abstract",
        pl.col("text").str.len_chars().alias("text_chars"),
        pl.Series("text_sha256", sha),
    ).sort("work_id")


def technology_table(sample: pl.DataFrame) -> pl.DataFrame:
    """Rows (work, query, group) where the group fires in the title or abstract."""
    parts = []
    for query, prefix in (("natcom", "q"), ("gate_v2", "g")):
        for code, group in TECH.items():
            if query == "natcom" and code == "umbrella":
                continue
            t, a = f"{prefix}_{code}_title", f"{prefix}_{code}_abstract"
            parts.append(
                sample.filter(pl.col(t) | pl.col(a)).select(
                    short_id("id"),
                    pl.lit(query).alias("query"),
                    pl.lit(group).alias("group"),
                    pl.col(t).alias("title"),
                    pl.col(a).alias("abstract"),
                )
            )
    return pl.concat(parts).sort("work_id", "query", "group")


def port_firings(docs: pl.DataFrame, mode: str) -> pl.DataFrame:
    """Run the port on (UT, text) and return firings keyed by work_id."""
    rows = detect(docs, mode=mode)
    return rows.select(
        short_id("UT"), "system", "sdg", pl.col("query_id").cast(pl.Int32), "features"
    )


def stored_corrected(interim: Path) -> pl.DataFrame:
    """The port's corrected-mode firings stored by open_alex_adt_gate."""
    return pl.read_parquet(interim / "recall_sdg_hits.parquet").select(
        short_id("id"), "system", "sdg", pl.col("query_id").cast(pl.Int32), "features"
    )


def check_corrected(stored: pl.DataFrame, sample: pl.DataFrame) -> int:
    """Check the stored corrected firings against a fresh run on a subset."""
    subset = sample.select("id", "text").sample(CHECK_SUBSET, seed=20261102)
    fresh = port_firings(subset.rename({"id": "UT"}), "corrected")
    ids = subset.select(short_id("id"))
    kept = stored.join(ids, on="work_id", how="semi")
    keys = ["work_id", "system", "sdg", "query_id", "features"]
    a, b = kept.sort(keys), fresh.sort(keys)
    assert a.equals(b), "stored corrected firings differ from a fresh run"
    return a.height


def llm_table(probe, interim: Path, ids: pl.Series) -> tuple[pl.DataFrame, list]:
    """Per (work, exercise, label): each model's score, answer, positive; consensus.

    Positive pairs come from the probe's `load_llm`, the consensus from its
    `combine` ("intersection"). Returns the table and the probe's checks.
    """
    probe.INTERIM = interim
    tables, checks = [], []
    for exercise in ("sdg", "adt"):
        positive, wide = {}, None
        for model in probe.MODELS:
            pos, c = probe.load_llm(exercise, model, ids)
            checks.append(c)
            positive[model] = pos
            p1 = pl.read_parquet(
                interim / f"recall_{exercise}_pass1_{model}.parquet",
                columns=["wos_id", "code", "score"],
            ).rename({"wos_id": "id", "score": f"{model}_pass1_score"})
            p2 = pl.read_parquet(
                interim / f"recall_{exercise}_pass2_{model}.parquet",
                columns=["wos_id", "code", "label"],
            ).rename({"wos_id": "id", "label": f"{model}_pass2_answer"})
            m = (
                p1.join(p2, on=["id", "code"], how="left")
                .join(
                    pos.with_columns(pl.lit(True).alias(f"{model}_positive")),
                    on=["id", "code"],
                    how="left",
                )
                .with_columns(pl.col(f"{model}_positive").fill_null(False))
            )
            wide = (
                m
                if wide is None
                else wide.join(m, on=["id", "code"], how="full", coalesce=True)
            )
        rules = probe.combine(positive)
        wide = (
            wide.join(
                rules["intersection"].with_columns(
                    pl.lit(True).alias("consensus_positive")
                ),
                on=["id", "code"],
                how="left",
            )
            .with_columns(pl.col("consensus_positive").fill_null(False))
            .with_columns(pl.lit(exercise).alias("exercise"))
        )
        tables.append(wide)
    out = pl.concat(tables).select(
        short_id("id"),
        "exercise",
        pl.col("code").alias("label"),
        *[
            c
            for m in probe.MODELS
            for c in (f"{m}_pass1_score", f"{m}_pass2_answer", f"{m}_positive")
        ],
        "consensus_positive",
    )
    assert out.select("work_id", "exercise", "label").is_duplicated().sum() == 0
    return out.sort("work_id", "exercise", "label"), checks


def assert_llm_counts(llm: pl.DataFrame, probe_dir: Path) -> None:
    """The positive works per rule and label must equal the probe's counts exactly."""
    expected = pl.read_csv(probe_dir / "llm_positive_counts.tsv", separator="\t")
    cols = {
        "qwen3": "qwen3_positive",
        "gemma": "gemma_positive",
        "intersection": "consensus_positive",
    }
    rows = []
    for exercise in ("sdg", "adt"):
        sub = llm.filter(pl.col("exercise") == exercise)
        for rule, col in cols.items():
            pos = sub.filter(pl.col(col))
            for label, n in pos.group_by("label").len().iter_rows():
                rows.append((exercise, rule, label, n))
            rows.append((exercise, rule, "any", pos["work_id"].n_unique()))
        union = sub.filter(pl.col("qwen3_positive") | pl.col("gemma_positive"))
        for label, n in union.group_by("label").len().iter_rows():
            rows.append((exercise, "union", label, n))
        rows.append((exercise, "union", "any", union["work_id"].n_unique()))
    got = pl.DataFrame(
        rows, schema=["exercise", "rule", "label", "works"], orient="row"
    )
    exp = expected.select("exercise", "rule", "label", "works").filter(
        pl.col("works") > 0
    )
    keys = ["exercise", "rule", "label"]
    assert got.sort(keys).equals(exp.sort(keys).cast(got.schema)), (
        "LLM counts differ from the probe"
    )


def r_complete(r_dir: Path, sizes: list[int]) -> list[str]:
    """Return what is missing from the R runs; empty when every chunk is there."""
    missing = []
    for mode in MODES:
        for i, n in enumerate(sizes):
            base = r_dir / mode / f"shard_{i}"
            want = math.ceil(n / CHUNK)
            have = len(list((base / "chunks").glob("chunk_*.parquet")))
            if have != want:
                missing.append(f"{mode}/shard_{i}: {have} of {want} chunks")
            if not (base / "coverage.parquet").exists():
                missing.append(
                    f"{mode}/shard_{i}: no coverage.parquet (run not finished)"
                )
    return missing


def r_firings(r_dir: Path, mode: str) -> pl.DataFrame:
    """Read every chunk of one R mode, keyed by work_id."""
    paths = sorted((r_dir / mode).glob("shard_*/chunks/chunk_*.parquet"))
    df = pl.concat([pl.read_parquet(p) for p in paths])
    return df.select(
        short_id("document"),
        "system",
        "sdg",
        pl.col("query_id").cast(pl.Int32),
        "features",
    )


def set_one(r_dir: Path, port: dict[str, pl.DataFrame], ids: pl.Series) -> pl.DataFrame:
    """Join R and port firings per mode; write firings and the agreement table."""
    keys = ["work_id", "mode", "system", "sdg", "query_id"]
    parts = []
    for short, mode in MODES.items():
        r = r_firings(r_dir, short).with_columns(pl.lit(mode).alias("mode"))
        assert r.select(keys).is_duplicated().sum() == 0, f"R {mode}: repeated firing"
        assert r["work_id"].is_in(ids.implode()).all(), (
            f"R {mode}: work outside the sample"
        )
        p = port[mode].with_columns(pl.lit(mode).alias("mode"))
        j = r.rename({"features": "features_r"}).join(
            p.rename({"features": "features_port"}), on=keys, how="full", coalesce=True
        )
        parts.append(j)
    f = (
        pl.concat(parts)
        .with_columns(
            pl.col("features_r").is_not_null().alias("in_r"),
            pl.col("features_port").is_not_null().alias("in_port"),
        )
        .with_columns(
            (
                pl.col("in_r")
                & pl.col("in_port")
                & (pl.col("features_r") == pl.col("features_port"))
            )
            .fill_null(False)
            .alias("features_equal")
        )
    )
    out = DATA / "r_vs_port"
    out.mkdir(parents=True, exist_ok=True)
    f.sort(keys).write_parquet(out / "firings.parquet", compression="zstd")
    agg = [
        pl.len().alias("firings"),
        (pl.col("in_r") & pl.col("in_port")).sum().alias("both"),
        (~pl.col("in_r")).sum().alias("port_only"),
        (~pl.col("in_port")).sum().alias("r_only"),
        pl.col("features_equal").sum().alias("features_equal"),
    ]
    table = pl.concat(
        [
            f.group_by("mode", "system").agg(agg),
            f.group_by("mode").agg(agg).with_columns(pl.lit("all").alias("system")),
        ],
        how="diagonal",
    ).sort("mode", "system")
    table.write_csv(out / "agreement.tsv", separator="\t")
    return table


def fill_readme(table: pl.DataFrame | None) -> None:
    """Write the agreement table, or a note that set 1 is pending, into README.md."""
    readme = ROOT / "README.md"
    text = readme.read_text()
    if table is None:
        body = (
            "Set 1 is not built yet: the R runs on all 300,001 works are not complete."
        )
    else:
        cols = [
            "mode",
            "system",
            "firings",
            "both",
            "port_only",
            "r_only",
            "features_equal",
        ]
        head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
        rows = "\n".join(
            "| "
            + " | ".join(f"{v:,}" if isinstance(v, int) else str(v) for v in r)
            + " |"
            for r in table.select(cols).iter_rows()
        )
        body = head + rows
    start, end = text.index(README_START), text.index(README_END)
    readme.write_text(
        text[: start + len(README_START)] + "\n" + body + "\n" + text[end:]
    )


def write_sums() -> None:
    """Write SHA256SUMS for every file under data/."""
    lines = []
    for p in sorted(DATA.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{digest}  {p.relative_to(DATA)}")
    (DATA / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Build every file of the record; set 1 only when the R runs are complete."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--interim",
        type=Path,
        default=Path("/home/knuxv/Codebase/open_alex_adt_gate/data/interim"),
    )
    ap.add_argument("--r-dir", type=Path)
    ap.add_argument("--skip-r", action="store_true", help="write set 2 alone")
    args = ap.parse_args()
    if not args.skip_r and args.r_dir is None:
        ap.error("--r-dir is required unless --skip-r is given")

    probe = load_probe()
    sample = load_sample(args.interim)
    ids_full = sample["id"]
    ids = sample.select(short_id("id"))["work_id"]

    # Refuse early, before any file is written, if set 1 is asked for and not ready.
    n, size = sample.height, math.ceil(sample.height / SHARDS)
    sizes = [min(size, n - i * size) for i in range(SHARDS)]
    if not args.skip_r:
        missing = r_complete(args.r_dir, sizes)
        if missing:
            sys.exit("R runs incomplete, set 1 not written:\n  " + "\n  ".join(missing))

    (DATA / "annotations").mkdir(parents=True, exist_ok=True)
    works_table(sample).write_parquet(DATA / "works.parquet", compression="zstd")
    technology_table(sample).write_parquet(
        DATA / "annotations/technology.parquet", compression="zstd"
    )

    corrected = stored_corrected(args.interim)
    checked = check_corrected(corrected, sample)
    print(
        f"corrected firings: stored equal fresh on {CHECK_SUBSET} works ({checked} firings)"
    )
    compatible = port_firings(
        sample.select(pl.col("id").alias("UT"), "text"), "compatible"
    )
    port = {"corrected": corrected, "compatible": compatible}
    sdg = pl.concat(
        [
            port[m].with_columns(pl.lit(m).alias("mode"))
            for m in ("corrected", "compatible")
        ]
    ).select("work_id", "mode", "system", "sdg", "query_id", "features")
    sdg.sort("work_id", "mode", "system", "sdg", "query_id").write_parquet(
        DATA / "annotations/sdg.parquet", compression="zstd"
    )

    llm, _checks = llm_table(probe, args.interim, ids_full)
    assert_llm_counts(llm, PROBE.parent)
    llm.write_parquet(DATA / "annotations/llm.parquet", compression="zstd")
    print("LLM positive counts equal the probe's")

    table = None
    if not args.skip_r:
        table = set_one(args.r_dir, port, ids)
        print(table)
    fill_readme(table)
    write_sums()


if __name__ == "__main__":
    main()
