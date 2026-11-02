-- Aggregate Google Meridian's geo-level demo data to a national weekly panel.
-- Revenue = conversions x revenue_per_conversion (KPI is conversions; Meridian
-- converts with a per-geo revenue-per-KPI series).
SELECT time::DATE AS week_start,
       count(DISTINCT geo)                                   AS n_geos,
       sum(conversions * revenue_per_conversion)              AS revenue,
       sum(conversions)                                       AS conversions,
       sum(Channel0_spend) AS spend_channel0, sum(Channel1_spend) AS spend_channel1,
       sum(Channel2_spend) AS spend_channel2, sum(Channel3_spend) AS spend_channel3,
       sum(Channel4_spend) AS spend_channel4,
       -- population-weighted controls
       sum(competitor_sales_control * population) / sum(population) AS competitor_sales,
       sum(sentiment_score_control * population) / sum(population)  AS sentiment,
       sum(Promo * population) / sum(population)                    AS promo,
       sum(Organic_channel0_impression)                              AS organic_impressions
FROM read_csv_auto('{{path}}')
GROUP BY ALL
ORDER BY week_start;
