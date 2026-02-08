#!/usr/bin/python3.12
"""
Generic constraint function generator from JSON alpha attribute.
Splits constraints by logical AND and generates Python functions.
"""
import json
import re
from typing import List
from sys import argv

def extract_variable_names(json_data: dict) -> List[str]:
    """
    Extract variable names from the 'variables' section of JSON.
    Only includes variables with 'interface' == 'knob' (input variables).
    
    Args:
        json_data: Parsed JSON dictionary
        
    Returns:
        List of variable label names
    """
    variables = json_data.get('variables', [])
    var_names = []
    
    for var in variables:
        # Only include input variables (knobs), not outputs
        if var.get('interface') in ['knob', 'slider', 'input']:
            var_names.append(var.get('label'))
    
    return var_names


def split_constraints_by_and(alpha_string: str) -> List[str]:
    """
    Split alpha string into individual constraints using logical AND.
    
    Args:
        alpha_string: The alpha constraint expression
        
    Returns:
        List of individual constraint strings
    """
    # Split by 'and' operator (case-insensitive, with word boundaries)
    constraints = re.split(r'\s+and\s+', alpha_string, flags=re.IGNORECASE)
    
    return [c.strip() for c in constraints if c.strip()]


def generate_constraint_function(constraint_num: int, constraint_expr: str, var_names: List[str]) -> str:
    """
    Generate a Python constraint function from a constraint expression.
    
    Args:
        constraint_num: Constraint number (for function naming)
        constraint_expr: Constraint expression as a string (valid Python boolean expression)
        var_names: List of variable names
        
    Returns:
        Python function code as string
    """
    # Generate function parameters
    params = ', '.join(var_names)
    
    # Generate function code
    func_code = f'''def constraint_C{constraint_num}({params}):
    """
    C{constraint_num}: {constraint_expr}
    Returns: True if constraint is satisfied, False otherwise
    """
    return {constraint_expr}'''
    
    return func_code


def generate_constraints_from_json(json_file: str) -> str:
    """
    Read JSON file and generate Python constraint functions from alpha attribute.
    
    Args:
        json_file: Path to JSON file
        
    Returns:
        String containing all Python constraint function definitions
    """
    # Read JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract variable names
    var_names = extract_variable_names(data)
    
    if not var_names:
        raise ValueError("No input variables found in JSON file")
    
    # Get alpha constraint
    alpha = data.get('alpha', '')
    
    if not alpha:
        raise ValueError("No 'alpha' attribute found in JSON file")
    
    # Split into individual constraints by AND
    constraints = split_constraints_by_and(alpha)
    
    # Generate header
    output = "# " + "=" * 76 + "\n"
    output += "# GENERATED CONSTRAINT FUNCTIONS FROM JSON ALPHA ATTRIBUTE\n"
    output += "# " + "=" * 76 + "\n"
    output += f"# Source: {json_file}\n"
    output += f"# Variables: {', '.join(var_names)}\n"
    output += f"# Alpha: {alpha}\n"
    output += f"# Number of constraints: {len(constraints)}\n"
    output += "# " + "=" * 76 + "\n\n"
    
    # Generate each constraint function
    for idx, constraint in enumerate(constraints, 1):
        func_code = generate_constraint_function(idx, constraint, var_names)
        output += func_code + "\n\n"
    
    return output


def generate_constraints(json_file: str = "bnh.json", output_file: str = "generated_constraints_claude.py"):
    
    print("=" * 80)
    print("GENERIC CONSTRAINT FUNCTION GENERATOR FROM JSON")
    print("=" * 80)
    print()
    
    
    try:
        # Generate constraint functions
        constraint_code = generate_constraints_from_json(json_file)
        
        # Print to console
        print(constraint_code)
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(constraint_code)
        
        print("=" * 80)
        print(f"✓ Constraints successfully generated and saved to '{output_file}'")
        print("=" * 80)
        
    except FileNotFoundError:
        print(f"ERROR: Could not find file '{json_file}'")
        return 1
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in file '{json_file}'")
        print(f"Details: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Get JSON file from command line or use default
    json_file = argv[1] if len(argv) > 1 else "bnh.json"
    output_file = argv[2] if len(argv) > 2 else "generated_constraints.py"
    exit(generate_constraints(json_file, output_file))
