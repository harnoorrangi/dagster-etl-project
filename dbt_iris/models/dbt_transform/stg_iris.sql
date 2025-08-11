{{ config(materialized='view') }}

select * from {{ source('dlt_main_dataset','iris') }}
