"""
Personal Fitness Tracker Dashboard
==================================
Log, analyze and visualize fitness activity data.

Concepts used:
    * Control Structures & Arrays (loops, conditionals, validation, NumPy arrays)
    * OOP                          (FitnessTracker class)
    * NumPy                        (averages, percentages, growth rate)
    * Pandas                       (load / clean / group / filter / new columns)
    * Matplotlib & Seaborn         (bar, line, pie and heatmap charts)

Run:
    python fitness_tracker.py
"""

import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "fitness_activities.csv")
CHART_DIR = os.path.join(BASE_DIR, "charts")

COL_DATE = "Date"
COL_TYPE = "Activity Type"
COL_DURATION = "Duration (Minutes)"
COL_CALORIES = "Calories Burned"
COL_CPM = "Calories per Minute"  # computed column
COLUMNS = [COL_DATE, COL_TYPE, COL_DURATION, COL_CALORIES]

DATE_FORMAT = "%Y-%m-%d"


# --------------------------------------------------------------------------
# Sample data generator (used if the CSV file does not exist)
# --------------------------------------------------------------------------
def generate_sample_csv(file_path=CSV_FILE, days=90, seed=42):
    """Generate a sample fitness_activities.csv with realistic random data."""
    rng = np.random.default_rng(seed)

    # activity: (min duration, max duration, calories per minute range)
    profiles = {
        "Running": (20, 60, (9.0, 13.0)),
        "Cycling": (30, 90, (6.5, 10.0)),
        "Yoga": (20, 60, (2.5, 4.5)),
        "Swimming": (20, 60, (7.0, 11.0)),
        "Walking": (20, 70, (3.5, 5.5)),
        "Gym": (30, 80, (5.0, 8.5)),
    }
    names = list(profiles.keys())
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)

    rows = []
    for offset in range(days):
        # roughly 70% of the days have an activity
        if rng.random() > 0.70:
            continue
        name = names[int(rng.integers(0, len(names)))]
        low, high, (cpm_low, cpm_high) = profiles[name]
        duration = int(rng.integers(low, high + 1))
        cal_per_min = rng.uniform(cpm_low, cpm_high)
        calories = int(round(duration * cal_per_min))
        date = (start + pd.Timedelta(days=offset)).strftime(DATE_FORMAT)
        rows.append([date, name, duration, calories])

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(file_path, index=False)
    return df


# --------------------------------------------------------------------------
# Main class
# --------------------------------------------------------------------------
class FitnessTracker:
    """Encapsulates all logic of the Personal Fitness Tracker Dashboard."""

    def __init__(self, csv_file=CSV_FILE):
        self.csv_file = csv_file
        self.df = pd.DataFrame(columns=COLUMNS)
        self.load_data()

    # ---------------------------- data loading ----------------------------
    def load_data(self):
        """Load the CSV into a DataFrame, create it if missing, and clean it."""
        if not os.path.exists(self.csv_file):
            print(f"[INFO] '{os.path.basename(self.csv_file)}' not found. "
                  "Generating sample data...")
            generate_sample_csv(self.csv_file)

        try:
            raw = pd.read_csv(self.csv_file)
        except pd.errors.EmptyDataError:
            print("[WARNING] CSV file is empty. Starting with an empty dataset.")
            raw = pd.DataFrame(columns=COLUMNS)
        except (pd.errors.ParserError, OSError, UnicodeDecodeError) as err:
            print(f"[ERROR] Could not read CSV file: {err}")
            raw = pd.DataFrame(columns=COLUMNS)

        # Make sure all the required columns exist
        missing = [col for col in COLUMNS if col not in raw.columns]
        if missing:
            print(f"[ERROR] CSV is missing columns: {missing}. Using empty dataset.")
            raw = pd.DataFrame(columns=COLUMNS)

        self.df = self._clean(raw[COLUMNS])

    def _clean(self, data):
        """Clean raw data: fix types, drop invalid / duplicate rows, add metrics."""
        df = data.copy()
        before = len(df)

        # Convert types (invalid values become NaN / NaT)
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], format=DATE_FORMAT, errors="coerce")
        df[COL_DURATION] = pd.to_numeric(df[COL_DURATION], errors="coerce")
        df[COL_CALORIES] = pd.to_numeric(df[COL_CALORIES], errors="coerce")

        # Remove rows with missing values
        df = df.dropna(subset=COLUMNS)

        # Standardise activity names
        df[COL_TYPE] = df[COL_TYPE].astype(str).str.strip().str.title()
        df = df[df[COL_TYPE] != ""]

        # Keep only positive numbers
        df = df[(df[COL_DURATION] > 0) & (df[COL_CALORIES] > 0)]

        # Remove exact duplicates, sort by date
        df = df.drop_duplicates()
        df = df.sort_values(COL_DATE).reset_index(drop=True)

        removed = before - len(df)
        if removed > 0:
            print(f"[INFO] Cleaning removed {removed} invalid/duplicate row(s).")

        return self._add_computed_columns(df)

    @staticmethod
    def _add_computed_columns(df):
        """Add the 'Calories per Minute' column."""
        df = df.copy()
        if df.empty:
            df[COL_CPM] = pd.Series(dtype=float)
        else:
            cpm = np.divide(df[COL_CALORIES].to_numpy(dtype=float),
                            df[COL_DURATION].to_numpy(dtype=float))
            df[COL_CPM] = np.round(cpm, 2)
        return df

    def save_data(self):
        """Write the dataset back to the CSV file (original 4 columns only)."""
        try:
            out = self.df[COLUMNS].copy()
            out[COL_DATE] = out[COL_DATE].dt.strftime(DATE_FORMAT)
            # store whole numbers as integers (55 instead of 55.0)
            for col in (COL_DURATION, COL_CALORIES):
                if (out[col] % 1 == 0).all():
                    out[col] = out[col].astype(int)
            out.to_csv(self.csv_file, index=False)
            return True
        except OSError as err:
            print(f"[ERROR] Could not save data: {err}")
            return False

    # --------------------------- activity logging -------------------------
    def log_activity(self, activity_type, duration, calories, date=None):
        """Validate and add a new activity. Returns True on success."""
        # ---- validation using control structures ----
        if not isinstance(activity_type, str) or not activity_type.strip():
            print("[ERROR] Activity type cannot be empty.")
            return False
        if not any(ch.isalpha() for ch in activity_type):
            print("[ERROR] Activity type must contain letters.")
            return False

        try:
            duration = float(duration)
            calories = float(calories)
        except (TypeError, ValueError):
            print("[ERROR] Duration and calories must be numbers.")
            return False

        if not np.isfinite(duration) or not np.isfinite(calories):
            print("[ERROR] Duration and calories must be finite numbers.")
            return False
        if duration <= 0:
            print("[ERROR] Duration must be a positive number.")
            return False
        if calories <= 0:
            print("[ERROR] Calories burned must be a positive number.")
            return False

        if date is None or str(date).strip() == "":
            date = datetime.today()
        else:
            try:
                date = datetime.strptime(str(date).strip(), DATE_FORMAT)
            except ValueError:
                print("[ERROR] Invalid date. Use format YYYY-MM-DD.")
                return False

        new_row = pd.DataFrame([{
            COL_DATE: pd.Timestamp(date).normalize(),
            COL_TYPE: activity_type.strip().title(),
            COL_DURATION: duration,
            COL_CALORIES: calories,
        }])

        base = self.df[COLUMNS]
        combined = new_row if base.empty else pd.concat([base, new_row], ignore_index=True)
        self.df = self._clean(combined)

        if self.save_data():
            print(f"[SUCCESS] Logged {activity_type.strip().title()} | "
                  f"{duration:g} min | {calories:g} kcal")
            return True
        return False

    # ------------------------------ metrics -------------------------------
    def calculate_metrics(self):
        """Return a dict of key metrics computed with NumPy."""
        if self.df.empty:
            return None

        durations = self.df[COL_DURATION].to_numpy(dtype=float)
        calories = self.df[COL_CALORIES].to_numpy(dtype=float)

        daily = self.df.groupby(COL_DATE)[COL_CALORIES].sum().to_numpy(dtype=float)

        # Percentage improvement: second half of days vs first half of days
        improvement = None
        if len(daily) >= 2:
            first_half, second_half = np.array_split(daily, 2)
            first_avg = float(np.mean(first_half))
            second_avg = float(np.mean(second_half))
            if first_avg > 0:
                improvement = (second_avg - first_avg) / first_avg * 100

        frequency = self.df[COL_TYPE].value_counts()
        percentage = np.round(frequency.to_numpy(dtype=float) / len(self.df) * 100, 1)

        return {
            "total_activities": int(len(self.df)),
            "total_calories": float(np.sum(calories)),
            "total_duration": float(np.sum(durations)),
            "avg_duration": float(np.mean(durations)),
            "avg_calories": float(np.mean(calories)),
            "max_calories": float(np.max(calories)),
            "daily_avg_calories": float(np.mean(daily)),
            "avg_cal_per_min": float(np.mean(self.df[COL_CPM].to_numpy(dtype=float))),
            "improvement_pct": improvement,
            "frequency": frequency,
            "percentage": dict(zip(frequency.index, percentage)),
            "active_days": int(len(daily)),
        }

    # ------------------------------ filtering -----------------------------
    def filter_activities(self, condition):
        """
        Filter activities.

        condition : dict with optional keys
            'activity_type' -> e.g. "Running"
            'start_date'    -> "YYYY-MM-DD"
            'end_date'      -> "YYYY-MM-DD"
        Returns a filtered DataFrame (empty if nothing matches / invalid input).
        """
        if self.df.empty:
            return self.df.copy()
        if not isinstance(condition, dict):
            print("[ERROR] Condition must be a dictionary.")
            return self.df.iloc[0:0].copy()

        result = self.df.copy()

        activity = condition.get("activity_type")
        if activity:
            result = result[result[COL_TYPE].str.lower() == str(activity).strip().lower()]

        for key, op in (("start_date", "ge"), ("end_date", "le")):
            value = condition.get(key)
            if value:
                try:
                    ts = pd.to_datetime(str(value).strip(), format=DATE_FORMAT)
                except ValueError:
                    print(f"[ERROR] Invalid {key}. Use format YYYY-MM-DD.")
                    return self.df.iloc[0:0].copy()
                result = result[result[COL_DATE] >= ts] if op == "ge" \
                    else result[result[COL_DATE] <= ts]

        return result.reset_index(drop=True)

    # -------------------------- grouping / trends -------------------------
    def group_by_activity(self):
        """Total / average duration and calories per activity type."""
        if self.df.empty:
            return None
        grouped = self.df.groupby(COL_TYPE).agg(
            Sessions=(COL_TYPE, "count"),
            Total_Minutes=(COL_DURATION, "sum"),
            Avg_Minutes=(COL_DURATION, "mean"),
            Total_Calories=(COL_CALORIES, "sum"),
            Avg_Cal_per_Min=(COL_CPM, "mean"),
        ).round(2)
        return grouped.sort_values("Total_Minutes", ascending=False)

    def weekly_trends(self):
        """Weekly totals of duration and calories."""
        if self.df.empty:
            return None
        weekly = (self.df.set_index(COL_DATE)[[COL_DURATION, COL_CALORIES]]
                  .resample("W").sum())
        weekly["Calories Change (%)"] = weekly[COL_CALORIES].pct_change() * 100
        weekly = weekly.replace([np.inf, -np.inf], np.nan).round(2)
        weekly.index = weekly.index.strftime(DATE_FORMAT)
        weekly.index.name = "Week Ending"
        return weekly

    # ------------------------------- report -------------------------------
    def generate_report(self):
        """Print a summary of the fitness data."""
        metrics = self.calculate_metrics()
        line = "=" * 56
        print("\n" + line)
        print("        PERSONAL FITNESS REPORT")
        print(line)

        if metrics is None:
            print("No activity data available. Log some activities first!")
            print(line)
            return

        print(f"Period               : {self.df[COL_DATE].min():%Y-%m-%d} "
              f"to {self.df[COL_DATE].max():%Y-%m-%d}")
        print(f"Total activities     : {metrics['total_activities']}")
        print(f"Active days          : {metrics['active_days']}")
        print(f"Total calories burned: {metrics['total_calories']:,.0f} kcal")
        print(f"Total time spent     : {metrics['total_duration']:,.0f} min "
              f"({metrics['total_duration'] / 60:.1f} hrs)")
        print(f"Average duration     : {metrics['avg_duration']:.1f} min / session")
        print(f"Average calories     : {metrics['avg_calories']:.1f} kcal / session")
        print(f"Daily avg calories   : {metrics['daily_avg_calories']:.1f} kcal / active day")
        print(f"Avg calories per min : {metrics['avg_cal_per_min']:.2f}")
        print(f"Best session         : {metrics['max_calories']:.0f} kcal")

        if metrics["improvement_pct"] is None:
            print("Improvement          : Not enough data")
        else:
            trend = "improved" if metrics["improvement_pct"] >= 0 else "dropped"
            print(f"Improvement          : {abs(metrics['improvement_pct']):.1f}% "
                  f"({trend}, 2nd half vs 1st half)")

        print("\nActivity frequency:")
        for name, count in metrics["frequency"].items():
            print(f"  - {name:<12}: {count:>3} sessions "
                  f"({metrics['percentage'][name]}%)")
        print(line)

    # ---------------------------- visualisation ---------------------------
    @staticmethod
    def _finish(fig, filename):
        """Save the figure to the charts folder and show it if possible."""
        try:
            os.makedirs(CHART_DIR, exist_ok=True)
            path = os.path.join(CHART_DIR, filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[INFO] Chart saved: {path}")
        except OSError as err:
            print(f"[WARNING] Could not save chart: {err}")

        if matplotlib.get_backend().lower() != "agg":
            plt.show()
        plt.close(fig)

    def _has_data(self, minimum=1):
        if len(self.df) < minimum:
            print(f"[WARNING] Need at least {minimum} activity record(s) for this chart.")
            return False
        return True

    def plot_bar_chart(self):
        """Bar chart: total time spent on each activity type."""
        if not self._has_data():
            return
        totals = (self.df.groupby(COL_TYPE)[COL_DURATION].sum()
                  .sort_values(ascending=False))
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(x=totals.index, y=totals.values, color="#4C72B0",
                    ax=ax, label="Total minutes")
        for i, value in enumerate(totals.values):
            ax.text(i, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_title("Total Time Spent on Each Activity", fontsize=14, fontweight="bold")
        ax.set_xlabel("Activity Type")
        ax.set_ylabel("Total Duration (Minutes)")
        ax.legend()
        fig.tight_layout()
        self._finish(fig, "bar_chart_time_per_activity.png")

    def plot_line_graph(self):
        """Line graph: calories burned over time."""
        if not self._has_data():
            return
        daily = self.df.groupby(COL_DATE)[COL_CALORIES].sum()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(daily.index, daily.values, marker="o", linewidth=1.5,
                color="#DD8452", label="Daily calories")
        if len(daily) >= 3:
            rolling = daily.rolling(window=7, min_periods=1).mean()
            ax.plot(rolling.index, rolling.values, linewidth=2.5,
                    color="#4C72B0", label="7-day average")
        ax.set_title("Calories Burned Over Time", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Calories Burned (kcal)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        self._finish(fig, "line_graph_calories_over_time.png")

    def plot_pie_chart(self):
        """Pie chart: percentage distribution of activities."""
        if not self._has_data():
            return
        counts = self.df[COL_TYPE].value_counts()
        colors = sns.color_palette("pastel", len(counts))
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(counts.values, labels=None, colors=colors, autopct="%1.1f%%",
               startangle=90, pctdistance=0.8,
               wedgeprops={"edgecolor": "white"})
        ax.legend(counts.index, title="Activity", loc="center left",
                  bbox_to_anchor=(1.0, 0.5))
        ax.set_title("Distribution of Activities", fontsize=14, fontweight="bold")
        ax.axis("equal")
        fig.tight_layout()
        self._finish(fig, "pie_chart_activity_distribution.png")

    def plot_heatmap(self):
        """Heatmap: correlation between duration and calories burned."""
        if not self._has_data(minimum=3):
            return
        cols = [COL_DURATION, COL_CALORIES, COL_CPM]
        corr = self.df[cols].corr()
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                    linewidths=0.5, cbar_kws={"label": "Correlation"}, ax=ax)
        ax.set_title("Correlation: Duration vs Calories Burned",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        self._finish(fig, "heatmap_correlation.png")

    def plot_all(self):
        """Generate every chart."""
        self.plot_bar_chart()
        self.plot_line_graph()
        self.plot_pie_chart()
        self.plot_heatmap()


# --------------------------------------------------------------------------
# Console helpers (control structures + input validation)
# --------------------------------------------------------------------------
def read_text(prompt):
    """input() wrapper that returns '' instead of crashing on EOF."""
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def read_positive_number(prompt, max_attempts=3):
    """Ask for a positive number; give the user a few attempts."""
    for _ in range(max_attempts):
        value = read_text(prompt)
        try:
            number = float(value)
            if np.isfinite(number) and number > 0:
                return number
            print("  -> Please enter a positive number.")
        except ValueError:
            print("  -> Invalid input. Please enter a number.")
    print("[ERROR] Too many invalid attempts.")
    return None


def print_table(df, title=None):
    """Pretty-print a DataFrame."""
    if title:
        print(f"\n{title}")
    if df is None or df.empty:
        print("No records to show.")
        return
    show = df.copy()
    if COL_DATE in show.columns:
        show[COL_DATE] = pd.to_datetime(show[COL_DATE]).dt.strftime(DATE_FORMAT)
    print(show.to_string(index=False))


def show_weekly(tracker):
    weekly = tracker.weekly_trends()
    print_table(None if weekly is None else weekly.reset_index(), "Weekly trends:")


def menu():
    print("\n" + "=" * 46)
    print("   PERSONAL FITNESS TRACKER DASHBOARD")
    print("=" * 46)
    print(" 1. Log a new activity")
    print(" 2. View all activities")
    print(" 3. Fitness report (metrics)")
    print(" 4. Filter activities")
    print(" 5. Activity summary (group by type)")
    print(" 6. Weekly trends")
    print(" 7. Bar chart  - time per activity")
    print(" 8. Line graph - calories over time")
    print(" 9. Pie chart  - activity distribution")
    print("10. Heatmap    - duration vs calories")
    print("11. Show ALL charts")
    print("12. Exit")
    print("-" * 46)


def handle_log(tracker):
    print("\n--- Log New Activity ---")
    activity = read_text("Activity type (e.g. Running, Yoga, Cycling): ")
    if not activity:
        print("[ERROR] Activity type cannot be empty.")
        return
    duration = read_positive_number("Duration (minutes): ")
    if duration is None:
        return
    calories = read_positive_number("Calories burned: ")
    if calories is None:
        return
    date = read_text("Date YYYY-MM-DD (press Enter for today): ")
    tracker.log_activity(activity, duration, calories, date or None)


def handle_filter(tracker):
    print("\n--- Filter Activities (press Enter to skip any field) ---")
    condition = {
        "activity_type": read_text("Activity type: "),
        "start_date": read_text("Start date (YYYY-MM-DD): "),
        "end_date": read_text("End date   (YYYY-MM-DD): "),
    }
    result = tracker.filter_activities(condition)
    print_table(result, f"Found {len(result)} matching record(s):")


def main():
    try:
        tracker = FitnessTracker()
    except Exception as err:  # last-resort guard so the app never crashes at start
        print(f"[FATAL] Could not start tracker: {err}")
        return

    actions = {
        "2": lambda: print_table(tracker.df, "All activities:"),
        "3": tracker.generate_report,
        "5": lambda: print_table(tracker.group_by_activity(), "Summary by activity:"),
        "6": lambda: show_weekly(tracker),
        "7": tracker.plot_bar_chart,
        "8": tracker.plot_line_graph,
        "9": tracker.plot_pie_chart,
        "10": tracker.plot_heatmap,
        "11": tracker.plot_all,
    }

    while True:
        menu()
        try:
            choice = input("Enter your choice: ").strip()
        except EOFError:
            print("\nGoodbye!")
            break
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if choice == "12":
            print("Goodbye! Stay fit!")
            break
        elif choice == "1":
            handle_log(tracker)
        elif choice == "4":
            handle_filter(tracker)
        elif choice in actions:
            actions[choice]()
        else:
            print("[ERROR] Invalid choice. Please enter a number from 1 to 12.")


if __name__ == "__main__":
    main()
