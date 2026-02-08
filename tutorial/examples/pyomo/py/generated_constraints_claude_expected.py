# ============================================================================
# GENERATED CONSTRAINT FUNCTIONS FROM JSON ALPHA ATTRIBUTE
# ============================================================================
# Source: /home/mdmitry/github/smlp_tutorial/tutorial/examples/pyomo/py/bnh.json
# Variables: X1, X2
# Alpha: (X1-5)*(X1-5)+X2*X2-25 and -(X1-8)*(X1-8)-(X2+3)*(X2+3)+7.7
# Number of constraints: 2
# ============================================================================

def constraint_C1(X1, X2):
    """
    C1: (X1-5)*(X1-5)+X2*X2-25
    Returns: True if constraint is satisfied, False otherwise
    """
    return (X1-5)*(X1-5)+X2*X2-25

def constraint_C2(X1, X2):
    """
    C2: -(X1-8)*(X1-8)-(X2+3)*(X2+3)+7.7
    Returns: True if constraint is satisfied, False otherwise
    """
    return -(X1-8)*(X1-8)-(X2+3)*(X2+3)+7.7

