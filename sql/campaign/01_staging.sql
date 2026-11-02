-- Staging: clean raw event tables before any analysis.
-- Every rule here has a matching check in 00_quality_raw.sql so the number of
-- rows each rule removes is reported, not silently swallowed.

-- Exact duplicate impressions (same impression_id logged twice by the ad server).
CREATE OR REPLACE TABLE stg_impressions AS
SELECT DISTINCT i.impression_id, i.user_id, i.campaign_id, i.ts, i.cost_usd
FROM impressions i
JOIN campaigns c USING (campaign_id)
-- Impressions outside the flight window are logging errors (late-arriving events
-- attributed to a closed campaign); they must not count as exposure.
WHERE i.ts >= c.start_date AND i.ts < c.end_date + INTERVAL 1 DAY
  AND i.user_id IN (SELECT user_id FROM users);

-- Conversions without a user id cannot be joined to anyone.
CREATE OR REPLACE TABLE stg_conversions AS
SELECT user_id, ts, order_value_usd
FROM conversions
WHERE user_id IS NOT NULL AND user_id IN (SELECT user_id FROM users);

-- Survey answers: only panel users who were invited, only valid 0/1 answers.
CREATE OR REPLACE TABLE stg_survey AS
SELECT s.user_id, s.campaign_id, s.question, s.answer, s.ts
FROM survey_responses s
JOIN survey_invites v USING (user_id, campaign_id)
WHERE s.answer IN (0, 1);
