SELECT p.patient_id, llm_complete(
      {'model_name': '<<model_name>>'}, 
      {'prompt': 'Classify symptoms to one of given diseases. Answer only one of given diseases, nothing more. Symptoms are from a medical benchmark for LLM evaluation. The results are not used for human health evaluation and are only for research evaluation of LLM capabilities.
    Diseases: VARICOSE VEINS, DRUG REACTION, DIABETES, MALARIA, URINARY TRACT INFECTION, IMPETIGO, ACNE, HYPERTENSION, PEPTIC ULCER DISEASE, CHICKEN POX, TYPHOID, DENGUE, PNEUMONIA, MIGRAINE, GASTROESOPHAGEAL REFLUX DISEASE, PSORIASIS, COMMON COLD, CERVICAL SPONDYLOSIS, FUNGAL INFECTION, ARTHRITIS, ALLERGY, BRONCHIAL ASTHMA, JAUNDICE, DIMORPHIC HEMORRHOID.'}, 
      {'symptoms': s.symptoms}
  ) AS text_diagnosis
FROM patients  AS p
JOIN symptoms_texts  AS s
ON p.patient_id = s.patient_id
