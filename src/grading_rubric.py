def classify_property_condition(house, yard, driveway, street):
    """
    Classifies residential exterior condition based on rubric scores.

    Parameters:
        house (int): 0–2
        yard (int): 0–2
        driveway (int): 0–2
        street (int): 0–2

    Returns:
        total_score (int)
        label (str)
    """

    # Validate inputs
    scores = [house, yard, driveway, street]
    for s in scores:
        if s not in [0, 1, 2]:
            raise ValueError("All scores must be 0, 1, or 2.")

    total_score = sum(scores)

    # Classification logic
    if total_score <= 3:
        label = "poor"
    elif total_score <= 6:
        label = "average"
    else:
        label = "well_maintained"

    return total_score, label