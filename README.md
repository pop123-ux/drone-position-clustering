# Drone Position Estimation with Clustering

Estimate hidden drone positions from spark observations using **K-Means clustering** and predict future positions using **linear regression**.

## Repository description

A beginner-friendly machine learning project that uses K-Means clustering to estimate drone positions from spark clouds, then applies linear regression to forecast future drone movement.

## Project overview

In this dataset, the drones are not directly visible. Instead, each row represents a spark detected around a drone at a specific timestamp.

The objective is to reconstruct the drone formation by:

1. Counting the number of distinct snapshots.
2. Identifying the number of drones.
3. Estimating drone positions at each observed timestamp.
4. Predicting drone positions at a future timestamp.

The key idea is simple:

```text
sparks form clusters
clusters represent drones
cluster centers estimate drone positions
linear regression predicts future movement
```

## Dataset

The dataset is stored in:

```text
data/dataset.csv
```

Each row represents one spark observation.

| Column | Meaning |
|---|---|
| `Name` | Snapshot identifier |
| `Timestamp` | Time when the snapshot was captured |
| `X` | Horizontal spark coordinate in feet |
| `Y` | Vertical spark coordinate in feet |

The dataset contains **31,250 rows** and **50 distinct snapshots**. Each snapshot contains **625 spark observations**.

## Methodology

### 1. Snapshot counting

The number of snapshots is calculated using the number of unique `Name` values.

```python
df["Name"].nunique()
```

### 2. Drone counting

By plotting one snapshot, the spark points clearly form **5 spatial groups**. Each group corresponds to one drone.

Since each snapshot has 625 sparks and the points split into 5 groups, each drone contributes roughly 125 sparks.

```text
625 sparks / 5 drones = 125 sparks per drone
```

### 3. Drone position estimation

For each timestamp, K-Means is applied to the spark coordinates:

```python
KMeans(n_clusters=5)
```

The cluster centers are used as estimated drone positions.

### 4. Future position prediction

The drones move in the same direction with constant speed, so their movement can be approximated using linear regression:

```text
x = a * time + b
y = c * time + d
```

A separate regression model is trained for `x` and `y` coordinates for each drone.

## Repository structure

```text
drone-position-clustering/
├── data/
│   └── dataset.csv
├── docs/
│   └── medium_article.md
├── images/
│   └── sample_snapshot_clusters.png
├── notebooks/
│   └── cod.ipynb
├── outputs/
│   └── submission.csv
├── src/
│   └── drone_clustering.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## How to run

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/drone-position-clustering.git
cd drone-position-clustering
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the solution:

```bash
python src/drone_clustering.py
```

This will generate:

```text
outputs/submission.csv
```

## Results

The solution produces a submission file with the required columns:

```text
id,subtaskID,answer
```

Main results:

| Task | Result |
|---|---:|
| Number of snapshots | 50 |
| Number of drones | 5 |

## Skills demonstrated

- Data cleaning with pandas
- Exploratory data analysis
- Unsupervised learning
- K-Means clustering
- Feature engineering with timestamps
- Linear regression
- CSV submission generation
- Portfolio-style machine learning project organization

## Medium article

A full project write-up is available in:

```text
docs/medium_article.md
```

## License

This project is released under the MIT License.
