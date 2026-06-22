SELECT p.patient_id, AI.GENERATE(
    FORMAT("""
    Classify symptoms to one of given diseases  from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities. Answer only one of given diseases, nothing more. 
    Diseases: VARICOSE VEINS, DRUG REACTION, DIABETES, MALARIA, URINARY TRACT INFECTION, IMPETIGO, ACNE, HYPERTENSION, PEPTIC ULCER DISEASE, CHICKEN POX, TYPHOID, DENGUE, PNEUMONIA, MIGRAINE, GASTROESOPHAGEAL REFLUX DISEASE, PSORIASIS, COMMON COLD, CERVICAL SPONDYLOSIS, FUNGAL INFECTION, ARTHRITIS, ALLERGY, BRONCHIAL ASTHMA, JAUNDICE, DIMORPHIC HEMORRHOID.
    Symptoms: %s
    """, s.symptoms), 
    connection_id => '<<connection>>',
    model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config": {"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' 
    <<other_params>>).result as text_diagnosis
FROM medical_dataset.patients  AS p
JOIN medical_dataset.symptoms_texts  AS s
ON p.patient_id = s.patient_id
