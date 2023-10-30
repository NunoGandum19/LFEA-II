import os
from read_lab_file import read_lab_file  # Import the reader function from your reader file

# Function to convert a list of pairs to a Mathematica list
def to_mathematica_list(pair_list):
    mathematica_list = "{" + ", ".join([f"{{{x}, {y}}}" for x, y in pair_list]) + "}"
    return mathematica_list

# Ask for the suffix for the input .lab file
suffix = input("Enter the suffix for the input file (e.g., 'varre'): ")

# Construct the file path using the specified suffix
input_file_path = f"./fabry-perot/5C_{suffix}.lab"

# Check if the input file exists before proceeding
if not os.path.isfile(input_file_path):
    print(f"The input file {input_file_path} was not found.")
else:
    # Read the pair lists using the reader file
    pair_lists = read_lab_file(input_file_path)

    # Convert each pair list to a Mathematica list and store them in a list
    mathematica_lists = [to_mathematica_list(pair_list) for pair_list in pair_lists]

    # Create the output file with the input file name included
    output_file_name = f"output_{suffix}_all_lists.txt"

    # Save the list of lists to the output file in the same directory as the script
    with open(output_file_name, 'w') as file:
        file.write("\n".join(mathematica_lists))

    print(f"All lists of pairs saved to {output_file_name} in Mathematica format.")
