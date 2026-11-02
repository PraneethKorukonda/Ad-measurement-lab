-- Outcome rates by assignment and exposure (descriptive).
SELECT treatment, exposure, count(*) AS n, avg(visit) AS visit_rate, avg(conversion) AS conversion_rate
FROM read_csv_auto('{{path}}')
GROUP BY ALL ORDER BY ALL;
