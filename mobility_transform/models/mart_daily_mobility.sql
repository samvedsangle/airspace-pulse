{{ config(materialized='table') }}

WITH raw_arrivals AS (
    SELECT * FROM read_parquet('../data/bronze_flights.parquet')
)
SELECT
    timestamp::DATE AS flight_date,
    destination_city,
    AVG(lat) AS center_lat,
    AVG(lon) AS center_lon,
    COUNT(DISTINCT flight_id) AS total_aircraft
FROM raw_arrivals
GROUP BY 1, 2