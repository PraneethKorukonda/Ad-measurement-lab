-- Reproducible random sample of the TREATMENT arm for the observational study.
-- Inside the treatment arm, exposure is not randomized: who gets shown an ad
-- depends on browsing behaviour, the same behaviour that predicts conversion.
SELECT * FROM (SELECT * FROM read_csv_auto('{{path}}') WHERE treatment = 1)
USING SAMPLE reservoir({{n}} ROWS) REPEATABLE ({{seed}});
