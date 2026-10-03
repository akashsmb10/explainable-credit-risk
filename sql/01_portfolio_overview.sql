SELECT split, COUNT(*) AS clients, SUM(event) AS events, AVG(event) AS event_rate,
       AVG(score) AS mean_score, SUM(LIMIT_BAL) AS credit_limit_ntd
FROM portfolio GROUP BY split ORDER BY split;
