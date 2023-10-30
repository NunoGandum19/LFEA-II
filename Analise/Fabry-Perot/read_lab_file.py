# read_lab_file.py

# Function to extract pairs as floats from a line
def extract_pairs(line):
    parts = line.strip().split("\t")
    if len(parts) >= 3:
        try:
            first_value = float(parts[-2])
            second_value = float(parts[-1])
            return (first_value, second_value)
        except ValueError:
            return None
    else:
        return None

# Function to read and extract pair lists from a .lab file
def read_lab_file(file_path):
    pair_lists = []  # List to store multiple pair lists
    current_lines = []

    try:
        # Open the file
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Iterate through the lines
        for line in lines[28:]:  # Start from line 29
            if "NAN" in line:
                if current_lines:
                    pairs = [extract_pairs(l) for l in current_lines]
                    pairs = [p for p in pairs if p is not None]
                    pair_lists.append(pairs)
                current_lines = []  # Reset current_lines
            else:
                current_lines.append(line)

        # Append the last pair list if it's not empty
        if current_lines:
            pairs = [extract_pairs(l) for l in current_lines]
            pairs = [p for p in pairs if p is not None]
            pair_lists.append(pairs)

        return pair_lists

    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
