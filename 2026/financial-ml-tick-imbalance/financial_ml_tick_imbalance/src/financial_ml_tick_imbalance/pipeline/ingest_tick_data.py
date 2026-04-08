from financial_ml_tick_imbalance.data.dukascopy_loader import(download_tick_data,)

def run_ingestion():
    df = download_tick_data(
        symbol="EURUSD",
        start=datetime(2026,1,1)
        end=datetime(2026,1,3)      
    )

    df.to_csv(
        "data/raw/dukascopy/EURUSD_2026_01",
        index=False,    
    )

if __name__== "__main__":
    run_ingestion()