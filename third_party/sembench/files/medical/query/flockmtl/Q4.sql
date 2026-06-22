SELECT AVG(p.age)
FROM patients  AS p
JOIN symptoms_texts  AS s
ON p.patient_id = s.patient_id
WHERE llm_filter(
      {'model_name': '<<model_name>>'}, 
      {'prompt': 'Patient with these symptoms has skin acne. Yes or no.Symptoms are from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities.'}, 
      {'symptoms': s.symptoms}
  );
