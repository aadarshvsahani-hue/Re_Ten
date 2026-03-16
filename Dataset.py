import argparse
import os
import numpy as np
import pandas as pd

def generate_dataset(n=20, seed=42):
    np.random.seed(seed)

    # Basic identifiers
    user_id = [f"user_{i+1}" for i in range(n)]

    # Demographics
    age = np.random.randint(18, 65, size=n)  # years
    height = np.random.normal(170, 8, size=n).round(1)  # cm

    # Goals (categorical)
    goals = np.random.choice(
        ["weight_loss", "muscle_gain", "endurance", "maintenance"],
        size=n,
        p=[0.3, 0.3, 0.2, 0.2],
    )

    # Training schedule
    week = np.random.randint(1, 13, size=n)  # e.g., week in a 12-week program
    workout_days = np.random.randint(1, 7, size=n)  # days per week they plan to work out
    # missed_days more likely small; allow up to 3 missed days typically
    missed_days = np.minimum(np.random.poisson(0.8, size=n), 6)

    # Session metrics
    avg_duration = (
        np.random.normal(45, 12, size=n) + workout_days * 1.5
    )  # minutes per session, influenced by workout_days
    avg_duration = np.clip(avg_duration, 10, 240).round(1)

    avg_intensity = np.clip(np.random.normal(6, 1.8, size=n), 1, 10).round(1)  # 1-10 scale

    # Training load (simple heuristic): workout_days * duration * intensity / scale
    training_load = (workout_days * avg_duration * avg_intensity / 10).round(1)

    # Sleep and recovery
    avg_sleep = np.clip(np.random.normal(7, 0.9, size=n), 3.5, 10).round(2)  # hours
    sleep_score = (
        ((avg_sleep - 3.5) / (10 - 3.5)) * 100 + np.random.normal(0, 6, size=n)
    )
    sleep_score = np.clip(sleep_score, 0, 100).round(1)

    # Fatigue, recovery, stress, motivation (0-10 scales)
    fatigue = np.clip(10 - (sleep_score / 10.0) + np.random.normal(0, 1.0, size=n), 0, 10).round(1)
    recovery = np.clip((sleep_score / 10.0) + np.random.normal(0, 1.0, size=n), 0, 10).round(1)
    stress_level = np.clip(np.random.normal(4.5, 2.2, size=n), 0, 10).round(1)
    motivation = np.clip(10 - missed_days + np.random.normal(0, 1.2, size=n), 0, 10).round(1)

    # Health events
    illness = np.random.choice([0, 1], size=n, p=[0.9, 0.1])  # 10% chance
    injury = np.random.choice([0, 1], size=n, p=[0.95, 0.05])  # 5% chance

    # Compute retention probability (combine several signals), then sample retained
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    m_mot = norm(motivation)
    m_missed = norm(missed_days)
    m_duration = norm(avg_duration)
    m_recovery = norm(recovery)
    m_sleep = norm(sleep_score)

    # Linear score (weights chosen to reflect plausible influence)
    score = (
        1.2 * m_mot
        - 1.6 * m_missed
        + 0.6 * m_duration
        + 1.0 * m_recovery
        - 2.0 * injury
        - 1.6 * illness
        + 0.4 * m_sleep
    )

    # Convert to probabilities via logistic; center by mean so ~half retained before randomness
    logits = score - score.mean()
    probs = 1 / (1 + np.exp(-logits))
    retained = (np.random.rand(n) < probs).astype(int)

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "age": age,
            "height": height,
            "goal": goals,
            "week": week,
            "workout_days": workout_days,
            "missed_days": missed_days,
            "avg_duration": avg_duration,
            "avg_intensity": avg_intensity,
            "training_load": training_load,
            "avg_sleep": avg_sleep,
            "sleep_score": sleep_score,
            "fatigue": fatigue,
            "recovery": recovery,
            "stress_level": stress_level,
            "motivation": motivation,
            "illness": illness,
            "injury": injury,
            "retained": retained,
        }
    )

    # Ensure columns match requested order exactly (as listed)
    cols_order = [
        "age",
        "height",
        "goal",
        "week",
        "workout_days",
        "missed_days",
        "avg_duration",
        "avg_intensity",
        "training_load",
        "avg_sleep",
        "sleep_score",
        "fatigue",
        "recovery",
        "stress_level",
        "motivation",
        "illness",
        "injury",
        "retained",
    ]
    # Place user_id first for convenience
    df = df[["user_id"] + cols_order]

    return df

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic retention dataset.")
    parser.add_argument("--n", type=int, default=20, help="Number of users (default 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default="reten_dataset.csv", help="Output CSV path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df = generate_dataset(n=args.n, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()

# ...existing code...
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic retention dataset.")
    parser.add_argument("--n", type=int, default=20, help="Number of users (default 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default="reten_dataset.csv", help="Output CSV path")
-    args = parser.parse_args()
+    # In environments like Jupyter/VSCode the process argv can include extra entries.
+    # Use parse_known_args() to ignore unknown args so the module is safe to run interactively.
+    args, _ = parser.parse_known_args()
# ...existing code...