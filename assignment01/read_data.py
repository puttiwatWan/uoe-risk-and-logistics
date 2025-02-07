# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:17:58 2025

@author: ksearle
"""
import re
import numpy as np
from collections import defaultdict


def parse_dat_file(file_path):
    parsed_data = defaultdict(dict)  # Dictionary to store variables and their indexed values
    current_variable = None
    buffer = ""  # Buffer to collect multiline values

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("!"):  # Skip comments and empty lines
                continue

            if ":" in line:  # New variable starts
                if current_variable:  # Save the previous variable
                    store_buffer_data(current_variable, buffer, parsed_data)
                # Extract variable name and initialize buffer
                current_variable, value = map(str.strip, line.split(":", 1))
                current_variable = re.sub(r"\(.*?\)", "", current_variable).strip()  # Base variable name
                buffer = value  # Start collecting data for this variable
            else:
                buffer += " " + line  # Continue collecting data for the current variable

        # Save the last variable
        if current_variable:
            store_buffer_data(current_variable, buffer, parsed_data)

    # Convert defaultdict to regular dictionary for simplicity
    return {k: dict(v) if isinstance(v, dict) else v for k, v in parsed_data.items()}


def store_buffer_data(variable_name, buffer, parsed_data):
    """
    Parse the buffer content into indexed data and store it.
    """
    # Remove any stray '[' from the buffer before processing
    buffer = buffer.lstrip('[').rstrip(']')

    # Check if the value contains an array or just a scalar
    if "(" in buffer:  # Handle multi-dimensional array values
        pattern = re.compile(r"\((.*?)\)\s*([^()]+)")
        matches = pattern.findall(buffer)

        for index, value_str in matches:
            # Convert index into tuple for multi-dimensional keys
            index_tuple = tuple(map(int, index.split()))
            # Parse values (try converting to numbers)
            values = parse_values(value_str)
            parsed_data[variable_name][index_tuple] = values
    elif any(c in buffer for c in "0123456789"):  # Handle one-dimensional arrays (no parentheses but contains numbers)
        values = parse_values(buffer)
        parsed_data[variable_name] = values
    else:
        # It's a scalar value, store it directly
        parsed_data[variable_name] = parse_values(buffer)


def parse_values(value_str):
    """
    Parse a string of space-separated values into a list of numbers or strings.
    """
    # Strip any stray closing brackets (e.g., from the last value in the array)
    value_str = value_str.rstrip(']')

    try:
        # Try to parse as floats
        return [float(x) for x in value_str.split()]
    except ValueError:
        # Fallback to string array if parsing fails
        return value_str.split()


# Example usage
file_path = "CaseStudyData.txt"  # Replace with the actual file path
parsed_data = parse_dat_file(file_path)

# Print parsed variables and their indexed values
for variable, value_dict in parsed_data.items():
    print(f"{variable}:")
