import pandas as pd
from sklearn.preprocessing import StandardScaler
from clustering import elbow_method, train_kmeans, plot_clusters

# Step 1: Load Data
df = pd.read_csv("../data/Mall_Customers.csv")
print("Data Loaded ")

# Step 2: Feature Selection
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Step 3: Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Find Optimal K
print("Running Elbow Method...")
elbow_method(X_scaled)

# Step 5: Train KMeans (K=5)
print("Training KMeans Model...")
model, labels = train_kmeans(X_scaled, n_clusters=5)
df['Cluster'] = labels

# Step 6: Plot Clusters
print("Plotting Clusters...")
plot_clusters(df, model, scaler)

# Step 7: Print Results
print("\nCluster Summary:")
print(df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean())
print("\nProject Completed \nCheck 'outputs' folder for graphs.")
