import os

# Specify the directory containing your notebook files
notebook_directory = './'

# List all files in the directory
notebooks = [file for file in os.listdir(notebook_directory) if file.endswith('.ipynb')]

# Loop through each notebook file and clear the output
for notebook in notebooks:
    notebook_path = os.path.join(notebook_directory, notebook)
    
    # Use the Jupyter CLI to clear the output
    os.system(f'jupyter nbconvert --clear-output --inplace {notebook_path}')
    
print("Output cleared for all notebook files.")