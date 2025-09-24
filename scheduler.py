#schedule.py
import pandas as pd
import random

def solve_schedule(avail_df):
    employees = avail_df['Employee'].tolist()
    shifts = avail_df.columns[2:].tolist()  # skip Employee & MaxHoursPerWeek
    max_hours = dict(zip(employees, avail_df['MaxHoursPerWeek']))

    # Initialize schedule
    schedule = pd.DataFrame(0, index=employees, columns=shifts)

    # Shuffle employees to reduce bias
    random.shuffle(employees)

    # Assign shifts
    for s in shifts:
        # Find available employees who haven't reached max_hours
        candidates = [e for e in employees 
                      if avail_df.loc[avail_df['Employee']==e, s].values[0] == 1
                      and schedule.loc[e].sum() < max_hours[e]]
        if candidates:
            # Pick the one with fewest assigned hours so far
            chosen = min(candidates, key=lambda e: schedule.loc[e].sum())
            schedule.loc[chosen, s] = 1

    return schedule


def swap_shift(schedule, emp1, emp2, shift, availability):
    """
    Swap logic:
    - schedule: one is 1, the other is 0 (not 0,0)
    - availability (preview): both must be 1 (จริงๆว่างทั้งคู่)
    """
    v1 = int(schedule.loc[emp1, shift])
    v2 = int(schedule.loc[emp2, shift])

    # ดึง availability ของ emp1, emp2
    a1 = int(availability.loc[availability['Employee'] == emp1, shift].values[0])
    a2 = int(availability.loc[availability['Employee'] == emp2, shift].values[0])

    # ✅ เงื่อนไข swap
    if v1 != v2 and a1 == 1 and a2 == 1:
        schedule.loc[emp1, shift], schedule.loc[emp2, shift] = v2, v1
        return True, schedule
    return False, schedule

