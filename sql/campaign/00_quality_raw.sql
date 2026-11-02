-- Data-quality checks on RAW tables. Each row: check name, failing rows, total rows.
-- Run before staging; the staging layer removes exactly these rows.
SELECT 'duplicate_impression_rows' AS check_name,
       (SELECT count(*) FROM impressions) - (SELECT count(*) FROM (SELECT DISTINCT * FROM impressions)) AS failing,
       (SELECT count(*) FROM impressions) AS total
UNION ALL
SELECT 'impressions_outside_flight',
       (SELECT count(*) FROM impressions i JOIN campaigns c USING (campaign_id)
         WHERE i.ts < c.start_date OR i.ts >= c.end_date + INTERVAL 1 DAY),
       (SELECT count(*) FROM impressions)
UNION ALL
SELECT 'impressions_unknown_user',
       (SELECT count(*) FROM impressions WHERE user_id NOT IN (SELECT user_id FROM users)),
       (SELECT count(*) FROM impressions)
UNION ALL
SELECT 'conversions_null_user',
       (SELECT count(*) FROM conversions WHERE user_id IS NULL),
       (SELECT count(*) FROM conversions)
UNION ALL
SELECT 'survey_from_uninvited_or_unknown',
       (SELECT count(*) FROM survey_responses s
         WHERE NOT EXISTS (SELECT 1 FROM survey_invites v
                           WHERE v.user_id = s.user_id AND v.campaign_id = s.campaign_id)),
       (SELECT count(*) FROM survey_responses)
UNION ALL
SELECT 'survey_answer_out_of_domain',
       (SELECT count(*) FROM survey_responses WHERE answer NOT IN (0, 1)),
       (SELECT count(*) FROM survey_responses)
UNION ALL
SELECT 'users_duplicate_id',
       (SELECT count(*) - count(DISTINCT user_id) FROM users),
       (SELECT count(*) FROM users)
UNION ALL
SELECT 'campaign_window_invalid',
       (SELECT count(*) FROM campaigns WHERE end_date < start_date),
       (SELECT count(*) FROM campaigns);
