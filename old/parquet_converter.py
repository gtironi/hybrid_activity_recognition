import polars as pl
import os

def polars_conversion(csv_path, parquet_path, has_header=True):
    print(f"Starting conversion on: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist")
        return

    try:
        dtypes_map = {
            "calfId": pl.String, 
            "behaviour": pl.String
        }
        
        # Prepare lazy reading
        q = pl.scan_csv(csv_path, has_header=has_header, schema_overrides=dtypes_map)
        
        # Transformation pipeline
        q = q.with_columns(
            pl.col("dateTime").str.to_datetime(),
            pl.col(pl.Float64).cast(pl.Float32)
        )
        
        # Specific optimization for AcTBeCalf
        if "AcTBeCalf" in csv_path:
            q = q.with_columns(
                pl.col("behaviour").cast(pl.Categorical),
                pl.col("calfId").cast(pl.Categorical)
            )

        # Save Parquet file
        q.sink_parquet(parquet_path, compression="snappy")
        
        print(f"File successfully saved at: {parquet_path}")
        
    except Exception as e:
        print(f"Error: {e}")

polars_conversion('AcTBeCalf.csv', 'AcTBeCalf.parquet')
polars_conversion('Time_Adj_Raw_Data.csv', 'Time_Adj_Raw_Data.parquet')
