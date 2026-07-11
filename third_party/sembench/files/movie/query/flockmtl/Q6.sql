SELECT r1.id as id, r1.reviewId as reviewId1, r2.reviewId as reviewId2
FROM reviews_2000  AS r1
JOIN reviews_2000  AS r2
ON r1.id = r2.id AND r1.reviewId <> r2.reviewId
WHERE r1.id = 'ant_man_and_the_wasp_quantumania' AND llm_filter(
      {'model_name': '<<model_name>>'}, 
      {'prompt': 'These two movie reviews express opposite sentiments - one is positive and the other is negative'}, 
      {'review1': r1.reviewText, 'review2': r2.reviewText}
  )
LIMIT 10;