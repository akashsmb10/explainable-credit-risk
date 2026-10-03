SELECT risk_band, COUNT(*) AS clients, SUM(LIMIT_BAL) AS observed_credit_limit_ntd,
       SUM(score * 0.45 * 100000) AS scenario_loss_ntd
FROM portfolio WHERE split='test' AND risk_band IN ('HIGH','VERY HIGH')
GROUP BY risk_band ORDER BY risk_band;
