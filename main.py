import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from scipy.stats import skew
import os

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

def load_data():
    try:
        df = pd.read_csv("data/dataset_original.csv")
        print("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        print("Error: dataset_original.csv not found.")
        return None

def clean_data(df):
    try:
        df = df.drop_duplicates()
        df = df.fillna(df.mean(numeric_only=True))
        if "Type" in df.columns:
            df = df[df["Type"] == "L"]
        df.to_csv("data/dataset_cleaned.csv", index=False)
        print("Cleaning complete. Unique filter applied.")
        return df
    except Exception as e:
        print(f"Cleaning Error: {e}")
        return df

def analyze_data(df):
    try:
        numerical = df.select_dtypes(include=np.number)
        analysis_cols = [c for c in numerical.columns if c not in ['UDI', 'Target', 'Machine failure']]
        print("\n--- IEEE SECTION VI: STATISTICAL RESULTS ---")
        for col in analysis_cols:
            data_array = numerical[col].values
            mean_val = np.mean(data_array)
            std_val = np.std(data_array)
            val_skew = skew(data_array)
            print(f"{col} | Mean: {mean_val:.2f} | Std Dev: {std_val:.2f} | Skewness: {val_skew:.2f}")
    except Exception as e:
        print(f"Analysis Error: {e}")

def create_visuals(df):
    try:
        if not os.path.exists('outputs'): os.makedirs('outputs')

        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Technical Correlation Matrix: Acoustic Sensor Features", pad=20)
        plt.tight_layout()
        plt.savefig("outputs/heatmap.png")
        plt.close()

        plt.figure(figsize=(12, 7))
        sensor_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]']
        sns.boxplot(data=df[sensor_cols])
        plt.title("Boxplot Analysis: Sensor Range & Outlier Identification", pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Physical Measurement Parameters")
        plt.ylabel("Measurement Amplitude")
        plt.tight_layout()
        plt.savefig("outputs/boxplot.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x="Torque [Nm]", hue="Machine failure", fill=True)
        plt.title("Probability Density: Torque Distribution (Normal vs. Failure)", pad=15)
        plt.xlabel("Applied Torque [Nm]")
        plt.ylabel("Kernel Density Estimate")
        plt.tight_layout()
        plt.savefig("outputs/kdeplot.png")
        plt.close()
        
        plt.figure(figsize=(10,6))
        plt.hist(df["Torque [Nm]"], bins=20)
        plt.title("Torque Distribution Histogram")
        plt.xlabel("Torque [Nm]")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig("outputs/histogram.png")
        plt.close()
        
        fig1, ax1 = plt.subplots()
        y_torque = df['Torque [Nm]'].iloc[:100].values
        line1, = ax1.plot([], [], color='red', lw=2)
        ax1.set_xlim(0, 100)
        ax1.set_ylim(y_torque.min()-5, y_torque.max()+5)
        ax1.set_title("Time-Series Telemetry: Torque Variability")
        ax1.set_xlabel("Time Step (T)")
        ax1.set_ylabel("Torque Magnitude [Nm]")
        plt.tight_layout()
        def update_torque(i):
            line1.set_data(range(i), y_torque[:i])
            return line1,
        ani1 = animation.FuncAnimation(fig1, update_torque, frames=100, blit=True)
        ani1.save('outputs/torque_telemetry.gif', writer='pillow')
        plt.close()

        fig2, ax2 = plt.subplots()
        y_speed = df['Rotational speed [rpm]'].iloc[:100].values
        line2, = ax2.plot([], [], color='blue', lw=2)
        ax2.set_xlim(0, 100)
        ax2.set_ylim(y_speed.min()-50, y_speed.max()+50)
        ax2.set_title("Time-Series Telemetry: Rotational Velocity")
        ax2.set_xlabel("Time Step (T)")
        ax2.set_ylabel("Speed [rpm]")
        plt.tight_layout()
        def update_speed(i):
            line2.set_data(range(i), y_speed[:i])
            return line2,
        ani2 = animation.FuncAnimation(fig2, update_speed, frames=100, blit=True)
        ani2.save('outputs/speed_telemetry.gif', writer='pillow')
        plt.close()

        print("Engineering visuals exported with finalized IEEE-standard naming.")
    except Exception as e:
        print(f"Visualization Error: {e}")

def main():
    print("Initializing Engineering Data Pipeline...")
    raw_df = load_data()
    if raw_df is not None:
        cleaned_df = clean_data(raw_df)
        analyze_data(cleaned_df)
        create_visuals(cleaned_df)
        print("\nPipeline execution complete.")

if __name__ == "__main__":
    main()