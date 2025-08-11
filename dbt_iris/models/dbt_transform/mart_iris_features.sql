{{ config(materialized='table') }}

select
  sepal_length as sepal_length,
  sepal_width as sepal_width,
  petal_length as petal_length,
  petal_width as petal_width,
  species as species,
  (petal_length / nullif(sepal_length, 0)) as petal_sepal_ratio,
  (petal_width / nullif(sepal_width, 0)) as petal_sepal_w_ratio
from {{ ref('stg_iris') }}
