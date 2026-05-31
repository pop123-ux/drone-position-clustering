# Detecting Drone Positions from Spark Observations Using Clustering and Linear Regression

When working with real-world data, the object we want to track is not always directly visible. Sometimes, we only observe indirect signals around it.

In this project, I worked with a dataset from a simulated drone light show. The drones were moving together in a vertical plane, but the dataset did not contain the actual drone positions. Instead, it contained the positions of colored sparks emitted around each drone at different moments in time.

The challenge was to estimate how many drones were present, where they were located at each timestamp, and where they would move in the future.

## Problem intuition

Each row in the dataset represents one spark, not one drone. Since every drone emits sparks around itself, the sparks naturally form groups.

That gives us the main idea:

```text
one group of sparks = one drone
center of that group = estimated drone position
```

This is a perfect use case for clustering.

## Tools used

The solution uses:

- Python
- pandas
- scikit-learn
- K-Means clustering
- Linear regression

## Dataset structure

The dataset contains four columns:

| Column | Meaning |
|---|---|
| Name | Snapshot identifier |
| Timestamp | Moment when the snapshot was captured |
| X | Horizontal spark coordinate |
| Y | Vertical spark coordinate |

The first step is loading the data and converting timestamps into real datetime values.

```python
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


df = pd.read_csv("data/dataset.csv")

df["datetime"] = pd.to_datetime(
    df["Timestamp"],
    format="%b %d, %Y, %I:%M:%S %p"
)

df = df.sort_values("datetime")
```

## Counting snapshots

Each snapshot has a name, so counting unique snapshot names gives the number of distinct observations.

```python
number_of_snapshots = df["Name"].nunique()
```

In this dataset, the answer is:

```text
50 snapshots
```

## Finding the number of drones

The number of drones is not directly provided in the dataset. To infer it, I plotted the spark points from one snapshot.

The points formed five clear groups. Since each group is created by sparks around one drone, the number of drones is:

```text
5 drones
```

Each snapshot contains 625 spark observations, which also supports this conclusion:

```text
625 sparks / 5 drones = 125 sparks per drone
```

## Estimating drone positions with K-Means

K-Means is an unsupervised learning algorithm that separates data points into groups.

In this project, I applied K-Means separately for each timestamp.

```python
number_of_drones = 5
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
```

The most important line is:

```python
centers = model.cluster_centers_
```

These centers are the estimated drone positions.

## Predicting future movement

The problem states that the drones continue moving in the same direction and with the same speed. That means the movement is approximately linear.

For each drone, I used linear regression to learn:

```text
x = a * time + b
y = c * time + d
```

First, I converted timestamps into seconds from the first snapshot.

```python
start_time = df["datetime"].min()

centers_df["time_seconds"] = (
    centers_df["datetime"] - start_time
).dt.total_seconds()
```

Then I predicted the future position.

```python
future_timestamp = "Jan 26, 2026, 12:08:20 AM"
future_datetime = pd.to_datetime(
    future_timestamp,
    format="%b %d, %Y, %I:%M:%S %p"
)

future_seconds = (future_datetime - start_time).total_seconds()
```

For each drone, I trained one regression model for `x` and one for `y`.

```python
for drone_id in range(number_of_drones):
    drone_data = centers_df[centers_df["droneID"] == drone_id]

    x_time = drone_data[["time_seconds"]]

    model_x = LinearRegression()
    model_y = LinearRegression()

    model_x.fit(x_time, drone_data["x"])
    model_y.fit(x_time, drone_data["y"])

    future_x = model_x.predict([[future_seconds]])[0]
    future_y = model_y.predict([[future_seconds]])[0]
```

## Final result

The final output is a CSV file with the required columns:

```text
id,subtaskID,answer
```

The full solution generates this file automatically.

## What I learned

This project was useful because it combines two simple machine learning techniques in a practical way.

K-Means helped identify hidden object positions from groups of observed points. Linear regression helped model movement over time and predict future coordinates.

The most important lesson is that machine learning is not only about complex models. Sometimes, the strongest solution comes from understanding the structure of the data.

## Conclusion

In this project, I estimated drone positions from spark observations. The drones were not directly visible, but their sparks formed spatial clusters. By using K-Means, I estimated drone positions from cluster centers. Then I used linear regression to predict future locations.

This makes the project a clear example of how unsupervised learning and regression can work together in a real data workflow.
