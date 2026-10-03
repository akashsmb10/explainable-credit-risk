SELECT split, SEX, COUNT(*) AS clients, AVG(event) AS event_rate,
       AVG(score) AS mean_score, AVG(POWER(score-event,2)) AS brier
FROM portfolio GROUP BY split, SEX ORDER BY split, SEX;
