{{ config(materialized='table') }}

select
  sepal_length as sepallength,
  sepal_width as sepalwidth,
  petal_length as petallength,
  petal_width as petalwidth,
  species as species,
  (petallength / nullif(sepallength, 0)) as petal_sepal_ratio,
  (petalwidth  / nullif(sepalwidth,  0)) as petal_sepal_w_ratio
from {{ ref('stg_iris') }}
