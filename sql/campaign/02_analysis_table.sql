-- One row per user: pre-period features, exposure, outcomes, survey status.
-- Features are computed strictly from data *before* campaign start, so that
-- nothing on the right-hand side can be affected by the ads (no post-treatment
-- controls).

CREATE OR REPLACE TABLE eligible AS
SELECT u.user_id, a.campaign_id, c.start_date, c.end_date, c.format, c.objective,
       c.frequency_cap, c.creative_score
FROM users u
JOIN user_campaign a USING (user_id)
JOIN campaigns c USING (campaign_id);

CREATE OR REPLACE TABLE features AS
SELECT e.user_id,
       ln(1 + u.sessions_30d)                              AS log_sessions,
       u.affinity,
       ln(u.affinity / (1 - u.affinity))                   AS affinity_logit,
       u.income_score                                      AS income,
       u.age_group, u.region, u.device,
       -- prior purchaser: any conversion in the 90 days before launch
       CAST(EXISTS (SELECT 1 FROM stg_conversions v
                    WHERE v.user_id = e.user_id
                      AND v.ts >= e.start_date - INTERVAL 90 DAY
                      AND v.ts <  e.start_date) AS INTEGER) AS prior_purchaser
FROM eligible e JOIN users u USING (user_id);

CREATE OR REPLACE TABLE exposure AS
SELECT e.user_id,
       count(i.impression_id)              AS frequency,
       CAST(count(i.impression_id) > 0 AS INTEGER) AS exposed,
       min(i.ts)                           AS first_exposure_ts,
       coalesce(sum(i.cost_usd), 0)        AS media_cost_usd
FROM eligible e
LEFT JOIN stg_impressions i ON i.user_id = e.user_id AND i.campaign_id = e.campaign_id
GROUP BY e.user_id;

CREATE OR REPLACE TABLE outcomes AS
SELECT e.user_id,
       CAST(EXISTS (SELECT 1 FROM stg_conversions v
                    WHERE v.user_id = e.user_id
                      AND v.ts >= e.start_date
                      AND v.ts <  e.end_date + INTERVAL 1 DAY) AS INTEGER) AS converted,
       CAST(EXISTS (SELECT 1 FROM survey_invites s WHERE s.user_id = e.user_id) AS INTEGER) AS surveyed,
       max(CASE WHEN s.question = 'aware'  THEN s.answer END) AS aware,
       max(CASE WHEN s.question = 'intent' THEN s.answer END) AS intent
FROM eligible e
LEFT JOIN stg_survey s ON s.user_id = e.user_id
GROUP BY e.user_id, e.start_date, e.end_date;

CREATE OR REPLACE TABLE analysis_users AS
SELECT e.user_id, e.campaign_id, e.format, e.objective, e.frequency_cap, e.creative_score,
       f.age_group, f.region, f.device, f.log_sessions, f.affinity, f.affinity_logit, f.income,
       f.prior_purchaser,
       x.exposed, x.frequency, x.media_cost_usd,
       o.converted, o.surveyed,
       CAST(o.aware IS NOT NULL AS INTEGER) AS responded,
       o.aware, o.intent
FROM eligible e
JOIN features f USING (user_id)
JOIN exposure x USING (user_id)
JOIN outcomes o USING (user_id)
ORDER BY e.user_id;
