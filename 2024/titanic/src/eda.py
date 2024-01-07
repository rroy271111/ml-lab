def print_summary(df):
    print("      Dataset summary    ")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\nMissing Values:")
    print(df.isna().sum().sort_values(ascending=False))

def survival_by_class(df):
    if {"Survived","Pclass"}.issubset(df.columns):
        print("\nSurvival Rate by Passenger Class:")
        print(df.groupby("Pclass")["Survived"].mean().round(3))

