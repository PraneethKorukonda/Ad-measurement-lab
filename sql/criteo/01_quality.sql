-- Criteo Uplift v2.1 (Diemert et al. 2018): data-quality checks on the full file.
-- Expected design: treatment share 0.85; control users can never be exposed.
WITH src AS (SELECT * FROM read_csv_auto('{{path}}'))
SELECT 'rows' AS check_name, count(*)::DOUBLE AS value, NULL::DOUBLE AS expected FROM src
UNION ALL SELECT 'null_cells', (count(*) * 16 - (count(f0)+count(f1)+count(f2)+count(f3)+count(f4)+count(f5)
        +count(f6)+count(f7)+count(f8)+count(f9)+count(f10)+count(f11)+count(treatment)+count(conversion)
        +count(visit)+count(exposure)))::DOUBLE, 0 FROM src
UNION ALL SELECT 'non_binary_flags', count(*) FILTER (WHERE treatment NOT IN (0,1) OR conversion NOT IN (0,1)
        OR visit NOT IN (0,1) OR exposure NOT IN (0,1))::DOUBLE, 0 FROM src
UNION ALL SELECT 'exposed_in_control', count(*) FILTER (WHERE treatment = 0 AND exposure = 1)::DOUBLE, 0 FROM src
UNION ALL SELECT 'conversion_without_visit', count(*) FILTER (WHERE conversion = 1 AND visit = 0)::DOUBLE, NULL FROM src
UNION ALL SELECT 'treatment_share', avg(treatment)::DOUBLE, 0.85 FROM src;
