import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


DATASET_PATH = "data/dataset.csv"
OUTPUT_PATH = "outputs/submission.csv"
TIMESTAMP_FORMAT = "%b %d, %Y, %I:%M:%S %p"
FUTURE_TIMESTAMP = "Jan 26, 2026, 12:08:20 AM"
NUMBER_OF_DRONES = 5


def main():
    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    # Convert timestamp text into real datetime values
    df["datetime"] = pd.to_datetime(
        df["Timestamp"],
        format=TIMESTAMP_FORMAT
    )

    df = df.sort_values("datetime")

    # Task 1: number of distinct snapshots
    number_of_snapshots = df["Name"].nunique()

    task1 = pd.DataFrame({
        "id": ["GLOBAL"],
        "subtaskID": ["task1"],
        "answer": [number_of_snapshots]
    })

    # Task 2: number of drones
    # The sparks form 5 clear spatial clusters in each snapshot.
    number_of_drones = NUMBER_OF_DRONES

    task2 = pd.DataFrame({
        "id": ["GLOBAL"],
        "subtaskID": ["task2"],
        "answer": [number_of_drones]
    })

    # Task 3: estimate drone positions using K-Means clustering
    centers_list = []

    for timestamp, group in df.groupby("Timestamp", sort=False):
        points = group[["X", "Y"]].values

        model = KMeans(
            n_clusters=number_of_drones,
            random_state=42,
            n_init=10
        )

        model.fit(points)
        centers = model.cluster_centers_

        for drone_id, center in enumerate(centers):
            centers_list.append({
                "Timestamp": timestamp,
                "datetime": group["datetime"].iloc[0],
                "droneID": drone_id,
                "x": center[0],
                "y": center[1]
            })

    centers_df = pd.DataFrame(centers_list)

    task3 = centers_df.copy()
    task3["id"] = task3["Timestamp"] + "|" + task3["droneID"].astype(str)
    task3["subtaskID"] = "task3"
    task3["answer"] = (
        task3["x"].round(6).astype(str)
        + "|"
        + task3["y"].round(6).astype(str)
    )
    task3 = task3[["id", "subtaskID", "answer"]]

    # Task 4: predict future drone positions using linear regression
    future_datetime = pd.to_datetime(
        FUTURE_TIMESTAMP,
        format=TIMESTAMP_FORMAT
    )

    start_time = df["datetime"].min()

    centers_df["time_seconds"] = (
        centers_df["datetime"] - start_time
    ).dt.total_seconds()

    future_seconds = (future_datetime - start_time).total_seconds()

    future_rows = []

    for drone_id in range(number_of_drones):
        drone_data = centers_df[centers_df["droneID"] == drone_id]

        x_time = drone_data[["time_seconds"]]

        model_x = LinearRegression()
        model_y = LinearRegression()

        model_x.fit(x_time, drone_data["x"])
        model_y.fit(x_time, drone_data["y"])

        future_x = model_x.predict([[future_seconds]])[0]
        future_y = model_y.predict([[future_seconds]])[0]

        future_rows.append({
            "id": FUTURE_TIMESTAMP + "|" + str(drone_id),
            "subtaskID": "task4",
            "answer": str(round(future_x, 6)) + "|" + str(round(future_y, 6))
        })

    task4 = pd.DataFrame(future_rows)

    # Build final submission
    submission = pd.concat(
        [task1, task2, task3, task4],
        ignore_index=True
    )

    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved submission to {OUTPUT_PATH}")
    print(f"Task 1 - snapshots: {number_of_snapshots}")
    print(f"Task 2 - drones: {number_of_drones}")


if __name__ == "__main__":
    main()
