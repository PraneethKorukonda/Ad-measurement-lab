WITH src AS (SELECT * FROM read_csv_auto('{{path}}'))
SELECT 'rows' AS check_name, count(*)::DOUBLE AS value FROM src
UNION ALL SELECT 'geos', count(DISTINCT geo)::DOUBLE FROM src
UNION ALL SELECT 'weeks', count(DISTINCT time)::DOUBLE FROM src
UNION ALL SELECT 'geo_week_duplicates', (count(*) - count(DISTINCT (geo, time)))::DOUBLE FROM src
UNION ALL SELECT 'negative_spend', count(*) FILTER (WHERE least(Channel0_spend, Channel1_spend, Channel2_spend,
        Channel3_spend, Channel4_spend) < 0)::DOUBLE FROM src
UNION ALL SELECT 'zero_spend_geo_weeks_channel2', count(*) FILTER (WHERE Channel2_spend = 0)::DOUBLE FROM src
UNION ALL SELECT 'null_kpi', count(*) FILTER (WHERE conversions IS NULL)::DOUBLE FROM src;
