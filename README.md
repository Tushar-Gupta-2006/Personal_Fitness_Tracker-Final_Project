# Personal Fitness Tracker Dashboard

A Python-based dashboard to **log, analyze and visualize** fitness activity data.
Built using Control Structures, Arrays (NumPy), OOP, Pandas, Matplotlib and Seaborn.

## Features
- **Input & validation** – duration/calories must be positive numbers, date must be `YYYY-MM-DD`, activity name cannot be empty.
- **`FitnessTracker` class (OOP)** with methods:
  - `log_activity(activity_type, duration, calories, date=None)`
  - `calculate_metrics()`
  - `filter_activities(condition)` – by activity type and/or date range
  - `generate_report()`
- **NumPy** – averages, percentages, daily averages, % improvement (2nd half vs 1st half).
- **Pandas** – load/clean CSV, group by activity, weekly trends, computed column *Calories per Minute*.
- **Visualizations** (saved in `charts/`): bar chart, line graph, pie chart, correlation heatmap.
- **Edge cases handled** – missing/empty/corrupt CSV, invalid rows, duplicates, empty dataset.

## Project Structure
```
fitness_tracker.py        # main program
fitness_activities.csv    # sample dataset (Date, Activity Type, Duration (Minutes), Calories Burned)
charts/                   # generated charts (PNG)
README.md
```

## Setup
1. Install Python 3.9+
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn
   ```
3. Run:
   ```bash
   python fitness_tracker.py
   ```
If `fitness_activities.csv` is missing, a sample dataset is generated automatically.

## Menu Options
| Option | Action |
|---|---|
| 1 | Log a new activity |
| 2 | View all activities |
| 3 | Fitness report (metrics) |
| 4 | Filter by activity type / date range |
| 5 | Summary grouped by activity |
| 6 | Weekly trends |
| 7–10 | Bar chart / Line graph / Pie chart / Heatmap |
| 11 | Show all charts |
| 12 | Exit |

## Example Usage (in code)
```python
from fitness_tracker import FitnessTracker

tracker = FitnessTracker()
tracker.log_activity("Running", 30, 300, "2026-10-27")
print(tracker.calculate_metrics())
print(tracker.filter_activities({"activity_type": "Running", "start_date": "2026-10-01"}))
tracker.generate_report()
tracker.plot_all()
```
