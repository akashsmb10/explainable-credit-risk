WITH ranked AS (
  SELECT *, NTILE(10) OVER (ORDER BY score DESC, ID) AS risk_decile
  FROM portfolio WHERE split = 'test'
)
SELECT risk_decile, COUNT(*) AS clients, SUM(event) AS events,
       AVG(event) AS event_rate, AVG(score) AS mean_score
FROM ranked GROUP BY risk_decile ORDER BY risk_decile;
