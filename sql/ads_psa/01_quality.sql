-- Ads vs. PSA experiment (public "Marketing A/B testing" dataset).
WITH src AS (SELECT * FROM read_csv_auto('{{path}}'))
SELECT 'rows' AS check_name, count(*)::DOUBLE AS value FROM src
UNION ALL SELECT 'duplicate_user_ids', (count(*) - count(DISTINCT user_id))::DOUBLE FROM src
UNION ALL SELECT 'null_cells', (count(*)*6 - (count(user_id)+count(test_group)+count(converted)+count(total_ads)
        +count(most_ads_day)+count(most_ads_hour)))::DOUBLE FROM src
UNION ALL SELECT 'unknown_group', count(*) FILTER (WHERE test_group NOT IN ('ad','psa'))::DOUBLE FROM src
UNION ALL SELECT 'total_ads_below_1', count(*) FILTER (WHERE total_ads < 1)::DOUBLE FROM src
UNION ALL SELECT 'hour_out_of_range', count(*) FILTER (WHERE most_ads_hour NOT BETWEEN 0 AND 23)::DOUBLE FROM src
UNION ALL SELECT 'total_ads_p99', quantile_cont(total_ads, 0.99)::DOUBLE FROM src
UNION ALL SELECT 'total_ads_max', max(total_ads)::DOUBLE FROM src
UNION ALL SELECT 'ad_share', avg((test_group = 'ad')::INT)::DOUBLE FROM src;
