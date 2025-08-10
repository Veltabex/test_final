import pandas as pd
import numpy as np
import joblib
import glob
import os
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import resource # This module might provide basic resource usage info on Unix-like systems

# --- Configuration ---
# !!! UPDATE THESE PATHS to match your Raspberry Pi file system !!!
# Assuming your test data CSVs are in a folder named 'test_data' in your home directory or similar
TEST_DATA_FOLDER = '/home/pi/Desktop/iot/test_data/' # Example path - UPDATED!
# Update these paths to your SVM model and scaler files
MODEL_PATH = '/home/pi/Desktop/iot/model/iot_malware_svm_subset_model.pkl' # Example path - UPDATED!
SCALER_PATH = '/home/pi/Desktop/iot/model/iot_malware_svm_subset_scaler.pkl' # Example path - UPDATED!

# IMPORTANT: Ensure these features match the ones used to train your SVM model
FEATURE_COLUMNS = ['HH_jit_L0_1_mean', 'HH_jit_L0_01_mean', 'HpHp_L0_01_radius',
                   'H_L0_01_weight', 'MI_dir_L0_1_weight'] # Add all subset features used by your model

# --- Data Loading Function ---
def load_test_data_from_folder(folder_path, feature_columns):
    """
    Loads all CSV files from the specified folder and concatenates them.
    Assumes each file contains rows of network traffic features.
    Adds a dummy 'true_label' column for simulation purposes.
    In a real scenario, you need a way to get actual true labels.
    """
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        print(f"No CSV files found in {folder_path}")
        return pd.DataFrame(), pd.Series() # Return empty DataFrame and Series

    list_dfs = []
    list_true_labels = [] # To store simulated true labels

    print(f"Loading test data from {folder_path}...")
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            # Ensure the loaded data has the expected feature columns
            if list(df.columns) != feature_columns:
                print(f"Warning: Columns in {filename} do not match expected features. Attempting to select/reorder.")
                try:
                    df = df[feature_columns]
                except KeyError as e:
                    print(f"Error: Missing expected feature columns in {filename}: {e}. Skipping file.")
                    continue # Skip file if required columns are missing


            list_dfs.append(df)

            # --- Simulate True Labels ---
            # !!! IMPORTANT: Replace this with your actual logic to get true labels !!!
            # Example: If file name contains 'malware', assume all samples are malware (1)
            # If file name contains 'normal', assume all samples are normal (0)
            # Otherwise, default to 0 (normal)
            true_label = 0
            if 'malware' in filename.lower():
                true_label = 1
            elif 'normal' in filename.lower():
                true_label = 0
            else:
                 # Default or derive from another source
                 pass # Keep default 0, or implement your logic


            list_true_labels.extend([true_label] * len(df)) # Add labels for all rows in the file


            print(f"Loaded {filename} ({len(df)} samples), simulated label: {true_label}")

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if list_dfs:
        test_df = pd.concat(list_dfs, ignore_index=True)
        true_labels_series = pd.Series(list_true_labels, name='true_label')
        print(f"Loaded total {len(test_df)} samples for testing.")
        return test_df, true_labels_series
    else:
        return pd.DataFrame(), pd.Series()


# --- Main Script Execution ---
if __name__ == "__main__":
    print("Starting test script...")

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

    # --- 2. Load Test Data ---
    start_time_data = time.time()
    test_data_df, true_labels = load_test_data_from_folder(TEST_DATA_FOLDER, FEATURE_COLUMNS)
    end_time_data = time.time()
    data_load_time = end_time_data - start_time_data
    print(f"⏱️ Data loading time: {data_load_time:.4f} seconds")

    if test_data_df.empty:
        print("❌ No test data loaded. Cannot proceed with predictions.")
        exit() # Stop execution if no data

    # Check if simulated true labels match data length
    if len(test_data_df) != len(true_labels):
         print("Warning: Number of test data samples and simulated true labels do not match!")
         # Decide how to handle this - for plotting, they must match.
         # For this demo, we'll truncate if mismatch occurs, but you should fix your label source.
         min_len = min(len(test_data_df), len(true_labels))
         test_data_df = test_data_df.head(min_len)
         true_labels = true_labels.head(min_len)
         print(f"Adjusted to {min_len} samples/labels for comparison.")


    # --- 3. Make Predictions ---
    start_time_predict = time.time()

    try:
        # Scale the test data
        print("Scaling test data...")
        test_data_scaled = scaler.transform(test_data_df)

        # Make predictions
        print("Making predictions...")
        predictions = model.predict(test_data_scaled)

        print("✅ Predictions completed.")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        exit()

    end_time_predict = time.time()
    prediction_time = end_time_predict - start_time_predict
    print(f"⏱️ Prediction time for {len(test_data_df)} samples: {prediction_time:.4f} seconds")
    print(f"Average prediction time per sample: {(prediction_time / len(test_data_df)):.6f} seconds")


    # --- 4. Evaluate Performance (requires true labels) ---
    print("\n--- Performance Evaluation ---")
    if not true_labels.empty and len(true_labels) == len(predictions):
        try:
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions)
            recall = recall_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions)
            conf_matrix = confusion_matrix(true_labels, predictions)

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision (Malware): {precision:.4f}")
            print(f"Recall (Malware): {recall:.4f}")
            print(f"F1-Score (Malware): {f1:.4f}")
            print("\nConfusion Matrix:")
            print(conf_matrix)

            # Optional: Print classification report for more detail
            # print("\nClassification Report:")
            # print(classification_report(true_labels, predictions))


        except Exception as e:
            print(f"❌ Error calculating performance metrics: {e}")
            print("Performance evaluation skipped.")
            # Set metrics to None or default for plotting
            accuracy, precision, recall, f1, conf_matrix = None, None, None, None, None
    else:
        print("⚠️ Cannot evaluate performance: True labels not available or mismatch with predictions.")
        accuracy, precision, recall, f1, conf_matrix = None, None, None, None, None


    # --- 5. Report Basic Resource Usage (Placeholder) ---
    print("\n--- Resource Usage (Basic) ---")
    try:
        # Get resource usage (might not be available or detailed on all systems/environments)
        # This gives maximum resident set size (memory usage)
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"Peak Memory Usage (Max RSS): {mem_usage} KB") # Units might vary (KB, MB, etc.)

        # Getting CPU usage over time is more complex and usually requires monitoring tools
        print("CPU Usage: Monitoring requires external tools or specific libraries.")

    except Exception as e:
        print(f"❌ Could not get basic resource usage: {e}")
        print("Resource usage reporting skipped (requires 'resource' module, usually on Unix-like systems).")


    # --- 6. Plotting Results ---
    print("\n--- Plotting Results ---")

    if not true_labels.empty and len(true_labels) == len(predictions):
         try:
             # Create a DataFrame for plotting
             plot_df = pd.DataFrame({
                 'True Label': true_labels,
                 'Prediction': predictions,
                 'Sample Index': test_data_df.index # Use original index
             })

             # Simple plot of True vs. Predicted labels over sample index
             plt.figure(figsize=(12, 6))
             sns.scatterplot(data=plot_df, x='Sample Index', y='True Label', label='True Label (0=Normal, 1=Malware)', alpha=0.6)
             sns.scatterplot(data=plot_df, x='Sample Index', y='Prediction', label='Prediction (0=Normal, 1=Malware)', alpha=0.6, marker='x', color='red')
             plt.title('True Labels vs. Predictions Over Samples')
             plt.xlabel('Sample Index')
             plt.ylabel('Label (0=Normal, 1=Malware)')
             plt.yticks([0, 1])
             plt.legend()
             plt.grid(axis='y', linestyle='--')
             plt.show()

             # Plot Confusion Matrix (if available)
             if conf_matrix is not None:
                  plt.figure(figsize=(8, 6))
                  sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                              xticklabels=['Predicted Normal', 'Predicted Malware'],
                              yticklabels=['True Normal', 'True Malware'])
                  plt.title('Confusion Matrix')
                  plt.xlabel('Predicted Label')
                  plt.ylabel('True Label')
                  plt.show()


         except Exception as e:
             print(f"❌ Error generating plots: {e}")
             print("Plotting skipped.")

    else:
        print("⚠️ Plotting requires both test data and corresponding true labels.")


    print("\nScript finished.")