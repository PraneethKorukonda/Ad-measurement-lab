SELECT user_id, (test_group = 'ad')::TINYINT AS ad, converted::TINYINT AS converted, total_ads,
       most_ads_day, most_ads_hour,
       CASE WHEN total_ads <= 5 THEN '1-5' WHEN total_ads <= 15 THEN '6-15'
            WHEN total_ads <= 30 THEN '16-30' WHEN total_ads <= 60 THEN '31-60'
            WHEN total_ads <= 120 THEN '61-120' ELSE '121+' END AS ads_bin
FROM read_csv_auto('{{path}}');
