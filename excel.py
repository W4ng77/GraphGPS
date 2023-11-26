import os
import json
import pandas as pd

# Define the base directory where the dataset folders are located
base_dir = 'actor'  # Update this to the correct path

# Initialize a list to collect DataFrame rows
rows_list = []

# List all model combination folders in the base directory
model_combination_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Process each model combination directory
for model_dir in model_combination_dirs:
    # Create the row name by removing the first three fields segmented by '-'
    #for web,wn
    #row_name = '-'.join(model_dir.split('-')[3:])

    #for actor
    row_name = '-'.join(model_dir.split('-')[2:])
    
    # Paths to the best.json files
    test_json_path = os.path.join(base_dir, model_dir, 'agg', 'test', 'best.json')
    train_json_path = os.path.join(base_dir, model_dir, 'agg', 'train', 'best.json')
    
    # Initialize a dictionary to hold the extracted data
    data_dict = {'Row_Name': row_name}
    
    # Extract data from the test best.json
    if os.path.isfile(test_json_path):
        with open(test_json_path, 'r') as file:
            data = json.load(file)
            data_dict['Test_Accuracy'] = data.get('accuracy')
            data_dict['Test_Accuracy_Std'] = data.get('accuracy_std')
    
    # Extract data from the train best.json
    if os.path.isfile(train_json_path):
        with open(train_json_path, 'r') as file:
            data = json.load(file)
            data_dict['Train_Accuracy'] = data.get('accuracy')
            data_dict['Train_Accuracy_Std'] = data.get('accuracy_std')
    
    # Add the dictionary to the list of rows
    rows_list.append(data_dict)

# Create a DataFrame from the list of rows
df = pd.DataFrame(rows_list)

# Set the row name as the index
df.set_index('Row_Name', inplace=True)

# Output file path
output_file_path = base_dir+'_accuracy_data.xlsx'  # Update this to the correct path

# Write the DataFrame to an Excel file
df.to_excel(output_file_path, engine='openpyxl')  # Specifying 'openpyxl' engine if 'xlsxwriter' is not installed

# Return the path to the output Excel file
output_file_path
