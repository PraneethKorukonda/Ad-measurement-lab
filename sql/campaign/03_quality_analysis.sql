-- Checks on the ANALYSIS table. All must be zero for the table to be used.
SELECT 'analysis_duplicate_users' AS check_name,
       count(*) - count(DISTINCT user_id) AS failing, count(*) AS total FROM analysis_users
UNION ALL
SELECT 'frequency_above_cap',
       count(*) FILTER (WHERE frequency > frequency_cap), count(*) FROM analysis_users
UNION ALL
SELECT 'exposed_flag_inconsistent',
       count(*) FILTER (WHERE (frequency > 0) <> (exposed = 1)), count(*) FROM analysis_users
UNION ALL
SELECT 'response_without_invite',
       count(*) FILTER (WHERE responded = 1 AND surveyed = 0), count(*) FROM analysis_users
UNION ALL
SELECT 'binary_outcome_out_of_domain',
       count(*) FILTER (WHERE converted NOT IN (0, 1) OR (aware IS NOT NULL AND aware NOT IN (0, 1))
                          OR (intent IS NOT NULL AND intent NOT IN (0, 1))), count(*) FROM analysis_users
UNION ALL
SELECT 'affinity_out_of_unit_interval',
       count(*) FILTER (WHERE affinity <= 0 OR affinity >= 1), count(*) FROM analysis_users;
