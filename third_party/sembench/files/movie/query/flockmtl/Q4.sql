SELECT AVG(
    CASE WHEN
    llm_filter(
        {'model_name': '<<model_name>>'}, 
        {'prompt': 'Is it a positive review?'}, 
        {'review': r.reviewText}
    )
    THEN 1 ELSE 0
    END
) AS positive_reviews_ratio
FROM reviews_2000 AS r
WHERE id = 'taken_3';
