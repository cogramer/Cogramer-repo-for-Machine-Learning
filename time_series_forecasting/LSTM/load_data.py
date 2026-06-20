import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from pathlib import Path
tqdm.pandas()

db_path = Path(__file__).parent / "12_HoChiMinh.db"

table_name = "data_25_01_12_to_25_12_23"

start_date = "2025-01-12"
end_date = "2025-12-23"
columns_to_plot = ["temperature", "humidity", "wind_speed", "pressure", "cloudiness", "solar_radiance"]

query = f'''
    SELECT datetime, {", ".join(columns_to_plot)}
    FROM {table_name}
    WHERE datetime BETWEEN ? AND ?
    ORDER BY datetime;
'''


def load_data():
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    conn.close()

    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def plot_data(df):
    plt.figure()

    for col in columns_to_plot:
        plt.plot(df["datetime"], df[col], label=col)

    plt.xlabel("Datetime")
    plt.ylabel("Value(s)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    df = load_data()
    print(df.head(10))
    print(df.info())
    plot_data(df)