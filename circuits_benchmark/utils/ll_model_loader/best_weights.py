def get_best_weight(case_name: str, individual=False):
    if str(case_name) in ['3']:
        ind_weights = {
            'iit': 1.0,
            'strict': 10.0,
            'behavior': 1.0,
        }
    elif str(case_name) in ['18', '34', '35', '36', '37']:
        ind_weights = {
            'iit': 1.0,
            'strict': 1.0,
            'behavior': 1.0,
        }
    elif str(case_name) in ['21']:
        ind_weights = {
            'iit': 1.0,
            'strict': 0.5,
            'behavior': 1.0,
        }
    elif "ioi" in str(case_name):
        if individual:
            return {
                'iit': 1.0,
                'strict': 0.4,
                'behavior': 1.0,
            }
        return "100_100_40"
    else:
        ind_weights = {
            'iit': 1.0,
            'strict': 0.4,
            'behavior': 1.0,
        }

    if individual:
        return ind_weights
    return str(int(ind_weights['strict'] * 1000 + ind_weights['behavior'] * 100 + ind_weights['iit'] * 10))
