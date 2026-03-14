from __future__ import annotations

from pathlib import Path


def _path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def csv(path: Path) -> str:
    return f"read_csv_auto('{_path(path)}', header=true, sample_size=-1, ignore_errors=false)"


def parquet(path: Path) -> str:
    return f"read_parquet('{_path(path)}')"


def parquet_dir(path: Path) -> str:
    return f"read_parquet('{_path(path)}/**/*.parquet', hive_partitioning=true)"


def event_weight_case(
    weight_view: float,
    weight_addtocart: float,
    weight_transaction: float,
) -> str:
    return f"""
        CASE
            WHEN event = 'view' THEN {weight_view}
            WHEN event = 'addtocart' THEN {weight_addtocart}
            WHEN event = 'transaction' THEN {weight_transaction}
            ELSE 0.0
        END
    """


def events_clean_sql(events_csv: Path) -> str:
    return f"""
    SELECT
        to_timestamp(CAST(timestamp AS BIGINT) / 1000.0) AS event_time,
        CAST(to_timestamp(CAST(timestamp AS BIGINT) / 1000.0) AS DATE) AS event_date,
        CAST(visitorid AS BIGINT) AS user_id,
        CAST(itemid AS BIGINT) AS item_id,
        lower(trim(CAST(event AS VARCHAR))) AS event,
        CAST(transactionid AS BIGINT) AS transaction_id
    FROM {csv(events_csv)}
    WHERE
        timestamp IS NOT NULL
        AND visitorid IS NOT NULL
        AND itemid IS NOT NULL
        AND event IS NOT NULL
    """


def purchases_sql(events_clean_dir: Path) -> str:
    return f"""
    SELECT
        event_time,
        event_date,
        user_id,
        item_id,
        transaction_id
    FROM {parquet_dir(events_clean_dir)}
    WHERE
        event = 'transaction'
        AND transaction_id IS NOT NULL
    """


def items_sql(item_props_part1_csv: Path, item_props_part2_csv: Path) -> str:
    return f"""
    WITH item_properties AS (
        SELECT
            to_timestamp(CAST(timestamp AS BIGINT) / 1000.0) AS property_time,
            CAST(itemid AS BIGINT) AS item_id,
            lower(trim(CAST(property AS VARCHAR))) AS property_name,
            trim(CAST(value AS VARCHAR)) AS property_value
        FROM {csv(item_props_part1_csv)}

        UNION ALL

        SELECT
            to_timestamp(CAST(timestamp AS BIGINT) / 1000.0) AS property_time,
            CAST(itemid AS BIGINT) AS item_id,
            lower(trim(CAST(property AS VARCHAR))) AS property_name,
            trim(CAST(value AS VARCHAR)) AS property_value
        FROM {csv(item_props_part2_csv)}
    )
    SELECT
        item_id,
        CAST(property_value AS BIGINT) AS category_id,
        property_time AS category_time
    FROM (
        SELECT
            item_id,
            property_value,
            property_time,
            row_number() OVER (
                PARTITION BY item_id
                ORDER BY property_time DESC
            ) AS row_num
        FROM item_properties
        WHERE
            property_name = 'categoryid'
            AND try_cast(property_value AS BIGINT) IS NOT NULL
    )
    WHERE row_num = 1
    """


def categories_sql(category_tree_csv: Path) -> str:
    return f"""
    SELECT DISTINCT
        CAST(categoryid AS BIGINT) AS category_id,
        CAST(parentid AS BIGINT) AS parent_id
    FROM {csv(category_tree_csv)}
    WHERE categoryid IS NOT NULL
    """


def stats_sql(
    events_clean_dir: Path,
    purchases_parquet: Path,
    items_parquet: Path,
    categories_parquet: Path,
) -> str:
    events_source = parquet_dir(events_clean_dir)
    purchases_source = parquet(purchases_parquet)
    items_source = parquet(items_parquet)
    categories_source = parquet(categories_parquet)

    return f"""
    SELECT 'events_total' AS metric, COUNT(*) AS value
    FROM {events_source}

    UNION ALL
    SELECT 'users_total' AS metric, COUNT(DISTINCT user_id) AS value
    FROM {events_source}

    UNION ALL
    SELECT 'items_total' AS metric, COUNT(DISTINCT item_id) AS value
    FROM {events_source}

    UNION ALL
    SELECT 'views_total' AS metric, COUNT(*) AS value
    FROM {events_source}
    WHERE event = 'view'

    UNION ALL
    SELECT 'addtocarts_total' AS metric, COUNT(*) AS value
    FROM {events_source}
    WHERE event = 'addtocart'

    UNION ALL
    SELECT 'transactions_total' AS metric, COUNT(*) AS value
    FROM {events_source}
    WHERE event = 'transaction'

    UNION ALL
    SELECT 'purchases_total' AS metric, COUNT(*) AS value
    FROM {purchases_source}

    UNION ALL
    SELECT 'item_rows_total' AS metric, COUNT(*) AS value
    FROM {items_source}

    UNION ALL
    SELECT 'categories_total' AS metric, COUNT(*) AS value
    FROM {categories_source}

    ORDER BY metric
    """


def interactions_sql(
    events_clean_dir: Path,
    weight_view: float,
    weight_addtocart: float,
    weight_transaction: float,
) -> str:
    return f"""
    SELECT
        user_id,
        item_id,
        event_time,
        {event_weight_case(weight_view, weight_addtocart, weight_transaction)} AS weight
    FROM {parquet_dir(events_clean_dir)}
    WHERE event IN ('view', 'addtocart', 'transaction')
    """


def raw_addtocart_events_sql(events_clean_dir: Path) -> str:
    return f"""
    SELECT
        user_id,
        item_id,
        event_time
    FROM {parquet_dir(events_clean_dir)}
    WHERE event = 'addtocart'
    """


def popular_items_sql(
    events_clean_dir: Path,
    weight_view: float,
    weight_addtocart: float,
    weight_transaction: float,
) -> str:
    return f"""
    WITH weighted_events AS (
        SELECT
            item_id,
            {event_weight_case(weight_view, weight_addtocart, weight_transaction)} AS weight
        FROM {parquet_dir(events_clean_dir)}
        WHERE event IN ('view', 'addtocart', 'transaction')
    )
    SELECT
        item_id,
        SUM(weight) AS total_weight
    FROM weighted_events
    GROUP BY item_id
    ORDER BY total_weight DESC
    """