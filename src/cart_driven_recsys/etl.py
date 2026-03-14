from __future__ import annotations

import json
import shutil
from pathlib import Path
import duckdb
from cart_driven_recsys.config import cfg
from cart_driven_recsys import sql
import logging

logger = logging.getLogger(__name__)

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _connect() -> duckdb.DuckDBPyConnection:
    _ensure_dir(cfg.s.duckdb_path.parent)
    return duckdb.connect(str(cfg.s.duckdb_path))


def _write_parquet(select_sql: str, output_path: Path, partition_by: str | None = None) -> Path:
    _ensure_dir(output_path.parent)

    tmp = output_path.with_name(f"{output_path.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp) if tmp.is_dir() else tmp.unlink()

    con = _connect()
    try:
        partition_clause = f", PARTITION_BY ({partition_by})" if partition_by else ""
        con.execute(f"""
            COPY ({select_sql})
            TO '{str(tmp).replace(chr(92), '/')}' (FORMAT PARQUET, COMPRESSION ZSTD{partition_clause})
        """)
    finally:
        con.close()

    if output_path.exists():
        shutil.rmtree(output_path) if output_path.is_dir() else output_path.unlink()
    tmp.rename(output_path)
    return output_path


def build_events_clean() -> Path:
    return _write_parquet(
        select_sql=sql.events_clean_sql(cfg.events_csv),
        output_path=cfg.events_clean_dir,
        partition_by="event_date",
    )


def build_purchases() -> Path:
    return _write_parquet(
        select_sql=sql.purchases_sql(cfg.events_clean_dir),
        output_path=cfg.purchases_parquet,
    )


def build_items() -> Path:
    return _write_parquet(
        select_sql=sql.items_sql(cfg.item_props_csvs[0], cfg.item_props_csvs[1]),
        output_path=cfg.items_parquet,
    )


def build_categories() -> Path:
    return _write_parquet(
        select_sql=sql.categories_sql(cfg.category_tree_csv),
        output_path=cfg.categories_parquet,
    )


def build_stats() -> Path:
    con = _connect()
    try:
        rows = con.execute(
            sql.stats_sql(
                events_clean_dir=cfg.events_clean_dir,
                purchases_parquet=cfg.purchases_parquet,
                items_parquet=cfg.items_parquet,
                categories_parquet=cfg.categories_parquet,
            )
        ).fetchall()
    finally:
        con.close()

    stats = {metric: int(value) for metric, value in rows}

    _ensure_dir(cfg.stats_json.parent)
    tmp = cfg.stats_json.with_name(f"{cfg.stats_json.name}.tmp")
    tmp.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if cfg.stats_json.exists():
        cfg.stats_json.unlink()
    tmp.rename(cfg.stats_json)
    return cfg.stats_json


def write_success_flag() -> Path:
    _ensure_dir(cfg.success_flag.parent)
    tmp = cfg.success_flag.with_name(f"{cfg.success_flag.name}.tmp")
    tmp.write_text("", encoding="utf-8")
    if cfg.success_flag.exists():
        cfg.success_flag.unlink()
    tmp.rename(cfg.success_flag)
    return cfg.success_flag


def run_all() -> None:
    logger.info("etl: build events_clean")
    build_events_clean()

    logger.info("etl: build purchases")
    build_purchases()

    logger.info("etl: build items")
    build_items()

    logger.info("etl: build categories")
    build_categories()

    logger.info("etl: build stats")
    build_stats()

    write_success_flag()
    logger.info("etl: done")

if __name__ == "__main__":
    run_all()