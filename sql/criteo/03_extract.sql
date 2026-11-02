-- Columns needed for the RCT benchmark (all rows) as compact integers.
SELECT treatment::TINYINT AS z, exposure::TINYINT AS d, visit::TINYINT AS visit,
       conversion::TINYINT AS conversion
FROM read_csv_auto('{{path}}');
