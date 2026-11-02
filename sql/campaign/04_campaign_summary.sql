-- Campaign-level delivery summary (descriptive, not causal).
SELECT campaign_id, any_value(format) AS format, any_value(objective) AS objective,
       any_value(frequency_cap) AS frequency_cap,
       count(*)                                   AS eligible_users,
       sum(exposed)                               AS reached_users,
       avg(exposed)                               AS reach_rate,
       sum(frequency)                             AS impressions,
       sum(frequency) / nullif(sum(exposed), 0)   AS avg_frequency,
       sum(media_cost_usd)                        AS spend_usd,
       1000 * sum(media_cost_usd) / nullif(sum(frequency), 0) AS cpm_usd,
       avg(converted) FILTER (WHERE exposed = 1)  AS conv_rate_exposed,
       avg(converted) FILTER (WHERE exposed = 0)  AS conv_rate_unexposed,
       sum(responded)                             AS survey_responses
FROM analysis_users
GROUP BY campaign_id
ORDER BY campaign_id;
