SELECT smoking_history, COUNT(*) AS count 
FROM patients 
WHERE smoking_history = "Never" AND audio_diagnosis NOT IN ('none', 'normal') AND 
    x_ray_diagnosis NOT IN ('none', '00_normal')
GROUP BY smoking_history
ORDER BY count DESC;
