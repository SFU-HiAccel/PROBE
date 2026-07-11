SELECT DISTINCT cars.car_id
FROM cars, car_complaints
WHERE cars.car_id = car_complaints.car_id
AND NLfilter(car_complaints.summary, 'In the complaint, the car has some problems with engine / connected to engine. Complaint: {summary}.')
