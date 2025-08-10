# --- Raspberry Pi Stress Test and Accuracy Script ---

import pandas as pd
import numpy as np
import joblib
import glob
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import resource # For basic resource usage on Unix-like systems
import matplotlib.pyplot as plt # Import matplotlib for plotting
import seaborn as sns # Import seaborn for enhanced plotting

# --- Configuration ---
# !!! UPDATE THESE PATHS to match your Raspberry Pi file system !!!
TEST_DATA_FOLDER = '/home/pi/Desktop/iot/test_data/' # Example path - UPDATE THIS!
MODEL_PATH = '/home/pi/Desktop/iot/model/iot_malware_svm_subset_model.pkl' # Example path - UPDATE THIS!
SCALER_PATH = '/home/pi/Desktop/iot/model/iot_malware_svm_subset_scaler.pkl' # Example path - UPDATE THIS!
PLOTS_SAVE_FOLDER = '/home/pi/Desktop/iot/test_output/plots/' # Define folder to save plots - NEW!


# --- Data Loading Function ---
def load_test_data_with_labels(folder_path):
    """
    Loads all CSV files from the specified folder, assuming they contain
    both features and a 'label' column with true labels (0 for Benign, 1 for Malware).
    Returns the features (X) and true labels (y).
    """
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        print(f"No CSV files found in {folder_path}")
        return pd.DataFrame(), pd.Series() # Return empty DataFrame and Series

    list_dfs = []

    print(f"Loading test data with true labels from {folder_path}...")
    for filename in all_files:
        try:
            # Load the CSV file
            df = pd.read_csv(filename)

            # Ensure the 'label' column exists
            if 'label' not in df.columns:
                print(f"Warning: 'label' column not found in {filename}. Skipping file.")
                continue

            list_dfs.append(df)
            print(f"Loaded {filename} ({len(df)} samples)")

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if list_dfs:
        combined_df = pd.concat(list_dfs, ignore_index=True)
        print(f"\nLoaded total {len(combined_df)} samples for testing.")

        # Separate features (X) and labels (y)
        if 'label' in combined_df.columns:
            y_true = combined_df['label']
            X_test_raw = combined_df.drop(columns=['label'], errors='ignore') # Drop label, ignore if already dropped
            print("Separated features and true labels.")
            return X_test_raw, y_true
        else:
            print("Error: 'label' column not found in combined data after loading.")
            return pd.DataFrame(), pd.Series()

    else:
        return pd.DataFrame(), pd.Series()


# --- Main Script Execution ---
if __name__ == "__main__":
    print("Starting Raspberry Pi Stress Test and Accuracy Script...")

    # Create the plots save folder if it doesn't exist
    os.makedirs(PLOTS_SAVE_FOLDER, exist_ok=True)
    print(f"Ensured plots save folder exists: {PLOTS_SAVE_FOLDER}")


    # --- 1. Load Model and Scaler ---
    start_time_load = time.time()
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Successfully loaded model and scaler.")
    except FileNotFoundError as e:
        print(f"❌ Error: Model or scaler file not found: {e}")
        print("Please update the MODEL_PATH and SCALER_PATH variables to the correct locations on your Pi.")
        exit() # Stop execution if files are not found
    except Exception as e:
        print(f"❌ Error loading model or scaler: {e}")
        exit()
    end_time_load = time.time()
    load_time = end_time_load - start_time_load
    print(f"⏱️ Model and Scaler loading time: {load_time:.4f} seconds")


    # --- 2. Load Test Data with True Labels ---
    # This function assumes your test data CSVs contain a 'label' column (0 or 1)
    start_time_data = time.time()
    X_test_raw, y_true = load_test_data_with_labels(TEST_DATA_FOLDER)
    end_time_data = time.time()
    data_load_time = end_time_data - start_time_data
    print(f"⏱️ Data loading time: {data_load_time:.4f} seconds")

    if X_test_raw.empty or y_true.empty or len(X_test_raw) != len(y_true):
        print("❌ Test data or true labels not loaded correctly. Cannot proceed with predictions and evaluation.")
        print("Please ensure your test CSVs are in the specified folder and contain a 'label' column.")
        exit() # Stop execution if data is not loaded correctly

    print(f"Loaded {len(X_test_raw)} samples for testing.")

    # --- 3. Prepare Data for Prediction (Align and Scale) ---
    print("\nPreparing data for prediction (aligning and scaling)...")
    try:
        # Align columns with the scaler's expected features
        scaler_feature_names = scaler.feature_names_in_

        # Add missing columns to X_test_raw with default value (e.g., 0)
        missing_cols_test = set(scaler_feature_names) - set(X_test_raw.columns)
        for c in missing_cols_test:
            X_test_raw[c] = 0

        # Ensure the order of columns in X_test_raw is the same as in scaler_feature_names
        X_test_aligned = X_test_raw[scaler_feature_names]

        # Scale the aligned test data
        X_test_scaled = scaler.transform(X_test_aligned)
        print("✅ Test data scaled successfully.")

    except Exception as e:
        print(f"❌ Error during data preparation (alignment or scaling): {e}")
        print("Please verify the columns in your test data CSVs match the data used to train the scaler/model.")
        exit()


    # --- 4. Make Predictions (Stress Test Component) ---
    print("\nMaking predictions (stress test)...")
    start_time_predict = time.time()

    try:
        # Make predictions on the scaled test data
        predictions = model.predict(X_test_scaled)
        print("✅ Predictions completed.")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        exit()

    end_time_predict = time.time()
    prediction_time = end_time_predict - start_time_predict
    print(f"⏱️ Prediction time for {len(X_test_scaled)} samples: {prediction_time:.4f} seconds")
    if len(X_test_scaled) > 0:
        print(f"Average prediction time per sample: {(prediction_time / len(X_test_scaled)):.6f} seconds")

    # --- 5. Evaluate Performance (Accuracy and other metrics) ---
    print("\n--- Performance Evaluation ---")
    try:
        # Ensure true_labels and predictions have the same index for correct comparison
        y_true = y_true.reset_index(drop=True)
        predictions_series = pd.Series(predictions).reset_index(drop=True)

        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        conf_matrix = confusion_matrix(y_true, predictions)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Malware): {precision:.4f}")
        print(f"Recall (Malware): {recall:.4f}")
        print(f"F1-Score (Malware): {f1:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)

        # Optional: Print classification report
        try:
             from sklearn.metrics import classification_report
             print("\nClassification Report:")
             print(classification_report(y_true, predictions, target_names=['Benign', 'Malware'], zero_division=0))
        except ImportError:
            print("sklearn.metrics.classification_report not available.")
        except Exception as e:
             print(f"Error generating classification report: {e}")


    except Exception as e:
        print(f"❌ Error calculating performance metrics: {e}")
        print("Performance evaluation skipped.")
        conf_matrix = None # Ensure conf_matrix is None if calculation fails


    # --- 6. Report Basic Resource Usage ---
    print("\n--- Resource Usage (Basic) ---")
    try:
        # Get resource usage (might not be available or detailed on all systems/environments)
        # This gives maximum resident set size (memory usage)
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux, ru_maxrss is typically in KB.
        print(f"Peak Memory Usage (Max RSS): {mem_usage} KB")

        # Getting CPU usage over time is more complex and usually requires monitoring tools
        # For a simple snapshot, you could use psutil if installed:
        # import psutil
        # print(f"Current CPU Usage: {psutil.cpu_percent(interval=1)}%")
        print("CPU Usage: For detailed monitoring, consider tools like 'htop' or 'atop' on your Pi.")

    except NameError:
        print("❌ 'resource' module not found (usually available on Unix-like systems). Resource usage reporting skipped.")
    except Exception as e:
        print(f"❌ Could not get basic resource usage: {e}")
        print("Resource usage reporting skipped.")


    # --- 7. Plotting Results and Saving ---
    print("\n--- Generating and Saving Plots ---")

    # Plot predicted class distribution
    try:
        if len(predictions) > 0:
            predicted_classes = pd.Series(predictions).map({0: 'Benign', 1: 'Malware'})
            predicted_counts = predicted_classes.value_counts()

            plt.figure(figsize=(8, 6))
            ax = sns.barplot(x=predicted_counts.index, y=predicted_counts.values, palette='viridis')
            plt.title('Predicted Class Distribution')
            plt.xlabel('Predicted Class')
            plt.ylabel('Count')
            for container in ax.containers:
                ax.bar_label(container)
            plot1_filename = os.path.join(PLOTS_SAVE_FOLDER, 'predicted_class_distribution.png')
            plt.savefig(plot1_filename)
            print(f"✅ Saved predicted class distribution plot to {plot1_filename}")
            plt.close() # Close the plot figure

        else:
            print("No predictions made, skipping predicted class distribution plot.")

    except Exception as e:
        print(f"❌ Error generating or saving predicted class distribution plot: {e}")
        print("Predicted class distribution plotting skipped.")


    # Plot Confusion Matrix (if available)
    try:
        if conf_matrix is not None:
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Predicted Benign', 'Predicted Malware'],
                        yticklabels=['True Benign', 'True Malware'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

            plot2_filename = os.path.join(PLOTS_SAVE_FOLDER, 'confusion_matrix_plot.png')
            plt.savefig(plot2_filename)
            print(f"✅ Saved confusion matrix plot to {plot2_filename}")
            plt.close() # Close the plot figure

        else:
             print("Confusion matrix not available, skipping confusion matrix plot.")


    except Exception as e:
        print(f"❌ Error generating or saving confusion matrix plot: {e}")
        print("Confusion matrix plotting skipped.")


    print("\nScript finished.")