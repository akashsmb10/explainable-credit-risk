SELECT risk_band, COUNT(*) AS clients, AVG(score) AS mean_score,
       AVG(event) AS event_rate, SUM(LIMIT_BAL) AS credit_limit_ntd
FROM portfolio WHERE split = 'test' GROUP BY risk_band ORDER BY mean_score;
