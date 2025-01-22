import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nDataset Information:")
    df.info()

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())  # No missing values in the Iris dataset

except FileNotFoundError:
    print("Error: Dataset file not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Task 2: Basic Data Analysis
try:
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Grouping by species and calculating the mean
    print("\nMean of features by species:")
    print(df.groupby('species').mean())

    # Identifying patterns:
    print("\nObservations:")
    print("- Different species have distinct mean values for sepal and petal measurements.")
    print("- 'setosa' tends to have smaller petal lengths and widths compared to 'versicolor' and 'virginica'.")

except Exception as e:
    print(f"An error occurred during data analysis: {e}")

# Task 3: Data Visualization
try:
    # 1. Bar chart: Average sepal length per species
    plt.figure(figsize=(8, 6))
    sns.barplot(x='species', y='sepal length (cm)', data=df)
    plt.title('Average Sepal Length per Species')
    plt.xlabel('Species')
    plt.ylabel('Average Sepal Length (cm)')
    plt.show()

    # 2. Histogram: Sepal width distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['sepal width (cm)'], kde=True) #kde adds a kernel density estimate line
    plt.title('Distribution of Sepal Width')
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Frequency')
    plt.show()

    # 3. Scatter plot: Sepal length vs. Petal length
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
    plt.title('Sepal Length vs. Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend(title='Species')
    plt.show()

    # 4. Boxplot of petal width by species
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="species", y="petal width (cm)", data=df)
    plt.title('Petal Width Distribution by Species')
    plt.xlabel('Species')
    plt.ylabel('Petal Width (cm)')
    plt.show()


except Exception as e:
    print(f"An error occurred during visualization: {e}")