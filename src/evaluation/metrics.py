def improvement(selected, baseline):
    return float(selected) - float(baseline)


def gap_to_upper_bound(selected, upper_bound):
    return float(upper_bound) - float(selected)
