SELECT reviewId
FROM reviews_2000 AS r
WHERE llm_filter(
      {'model_name': '<<model_name>>'}, 
      {'prompt': 'The following movie review is clearly positive.'}, 
      {'review': r.reviewText}
  )
LIMIT 5;