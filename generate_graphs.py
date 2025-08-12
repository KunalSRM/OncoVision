# # import os
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # os.makedirs("output", exist_ok=True)

# # def justify_and_plot_breast_cancer():
# #     print("\n=== Breast Cancer Dataset ===")
# #     df = pd.read_csv("../data/breast_cancer.csv")
# #     print(f"Shape: {df.shape}")
# #     # Drop unnecessary column
# #     df = df.drop(columns=["Unnamed: 32", "id"])
# #     df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

# #     # Distribution of diagnosis
# #     plt.figure(figsize=(6,4))
# #     sns.countplot(x='diagnosis', data=df)
# #     plt.title("Breast Cancer Diagnosis Distribution (0=Benign,1=Malignant)")
# #     plt.savefig("output/breast_cancer_diagnosis_dist.png")
# #     plt.close()
    
# #     print("Count of diagnosis values:")
# #     print(df['diagnosis'].value_counts())
# #     print("This justifies the countplot: the taller bar corresponds to benign (0) which has higher count.")

# #     # Example: Compare radius_mean for malignant vs benign
# #     malignant_radius_mean = df[df['diagnosis'] == 1]['radius_mean'].mean()
# #     benign_radius_mean = df[df['diagnosis'] == 0]['radius_mean'].mean()
# #     print(f"Average radius_mean for malignant tumors: {malignant_radius_mean:.2f}")
# #     print(f"Average radius_mean for benign tumors: {benign_radius_mean:.2f}")
# #     print("This supports any radius-related plots that show malignant tumors tend to have higher radius_mean.")

# # def justify_and_plot_lung_cancer():
# #     print("\n=== Lung Cancer Dataset ===")
# #     df = pd.read_csv("../data/lung_cancer.csv")
    
# #     # Clean columns with trailing spaces if any
# #     df.columns = [col.strip() for col in df.columns]
    
# #     print(f"Shape: {df.shape}")
# #     print("Unique values before mapping:")
# #     for col in df.columns:
# #         print(f"{col}: {df[col].unique()[:5]} ...")
    
# #     # Map categorical values
# #     mapping_gender = {"M":0, "F":1}
# #     mapping_yes_no = {"No":0, "Yes":1}
# #     df["GENDER"] = df["GENDER"].map(mapping_gender)
    
# #     yes_no_cols = ["SMOKING","YELLOW_FINGERS","ANXIETY","PEER_PRESSURE","CHRONIC_DISEASE",
# #                    "FATIGUE","ALLERGY","WHEEZING","ALCOHOL_CONSUMING","COUGHING",
# #                    "SHORTNESS_OF_BREATH","SWALLOWING_DIFFICULTY","CHEST_PAIN","LUNG_CANCER"]
# #     for col in yes_no_cols:
# #         df[col] = df[col].map(mapping_yes_no)
    
# #     # Plot smoking vs lung cancer count
# #     plt.figure(figsize=(6,4))
# #     sns.countplot(x='SMOKING', hue='LUNG_CANCER', data=df)
# #     plt.title("Smoking vs Lung Cancer (0=No, 1=Yes)")
# #     plt.savefig("output/lung_smoking_vs_cancer.png")
# #     plt.close()
    
# #     print("\nCounts for Smoking and Lung Cancer combinations:")
# #     print(pd.crosstab(df['SMOKING'], df['LUNG_CANCER']))
# #     print("This table justifies the graph, showing lung cancer prevalence is higher among smokers (SMOKING=1).")

# #     # Check lung cancer rate among smokers and non-smokers
# #     lung_cancer_smokers = df[df['SMOKING'] == 1]['LUNG_CANCER'].mean()
# #     lung_cancer_nonsmokers = df[df['SMOKING'] == 0]['LUNG_CANCER'].mean()
# #     print(f"Proportion of lung cancer among smokers: {lung_cancer_smokers:.2f}")
# #     print(f"Proportion of lung cancer among non-smokers: {lung_cancer_nonsmokers:.2f}")
# #     print("Higher proportion in smokers supports the graph pattern.")

# # if __name__ == "__main__":
# #     print("Generating and justifying graphs...\n")
# #     justify_and_plot_breast_cancer()
# #     justify_and_plot_lung_cancer()
# #     print("\nAll graphs saved in 'graphs/output/'.")

# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set seaborn style
# sns.set(style="whitegrid")

# # Create output directory inside graphs folder
# os.makedirs("graphs/output", exist_ok=True)

# def generate_all_breast_cancer_graphs():
#     print("\n=== Breast Cancer Dataset Graphs ===")
#     df = pd.read_csv("../data/breast_cancer.csv")
#     df = df.drop(columns=["Unnamed: 32", "id"])
#     df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

#     # 1. Countplot of diagnosis
#     plt.figure(figsize=(6,4))
#     sns.countplot(x='diagnosis', data=df)
#     plt.title("Countplot: Diagnosis (0=Benign, 1=Malignant)")
#     plt.savefig("graphs/output/breast_countplot_diagnosis.png")
#     plt.close()

#     # 2. Pie chart for diagnosis distribution
#     plt.figure(figsize=(6,6))
#     df['diagnosis'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Benign', 'Malignant'], colors=['#66b3ff','#ff6666'])
#     plt.title("Pie Chart: Diagnosis Distribution")
#     plt.ylabel('')
#     plt.savefig("graphs/output/breast_piechart_diagnosis.png")
#     plt.close()

#     # 3. Histogram of radius_mean
#     plt.figure(figsize=(6,4))
#     plt.hist(df['radius_mean'], bins=30, color='skyblue')
#     plt.title("Histogram: Radius Mean")
#     plt.xlabel("Radius Mean")
#     plt.ylabel("Frequency")
#     plt.savefig("graphs/output/breast_histogram_radius_mean.png")
#     plt.close()

#     # 4. Distplot of texture_mean (deprecated distplot replaced with hist + kde)
#     plt.figure(figsize=(6,4))
#     sns.histplot(df['texture_mean'], kde=True, color='orange')
#     plt.title("Distplot: Texture Mean")
#     plt.xlabel("Texture Mean")
#     plt.savefig("graphs/output/breast_distplot_texture_mean.png")
#     plt.close()

#     # 5. Boxplot of perimeter_mean by diagnosis
#     plt.figure(figsize=(6,4))
#     sns.boxplot(x='diagnosis', y='perimeter_mean', data=df, palette='Set2')
#     plt.title("Boxplot: Perimeter Mean by Diagnosis")
#     plt.savefig("graphs/output/breast_boxplot_perimeter_mean.png")
#     plt.close()

#     # 6. Scatterplot radius_mean vs texture_mean colored by diagnosis
#     plt.figure(figsize=(6,4))
#     sns.scatterplot(x='radius_mean', y='texture_mean', hue='diagnosis', data=df, palette=['green','red'])
#     plt.title("Scatterplot: Radius Mean vs Texture Mean")
#     plt.savefig("graphs/output/breast_scatterplot_radius_texture.png")
#     plt.close()

#     # 7. Barplot of mean area_mean by diagnosis
#     plt.figure(figsize=(6,4))
#     sns.barplot(x='diagnosis', y='area_mean', data=df, palette='pastel')
#     plt.title("Barplot: Mean Area by Diagnosis")
#     plt.savefig("graphs/output/breast_barplot_area_mean.png")
#     plt.close()

#     # 8. Heatmap of correlation matrix
#     plt.figure(figsize=(12,10))
#     corr = df.corr()
#     sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f", cbar=True)
#     plt.title("Heatmap: Feature Correlation")
#     plt.savefig("graphs/output/breast_heatmap_corr.png")
#     plt.close()

#     # 9. Clustermap of correlation matrix (hierarchical clustering)
#     cg = sns.clustermap(corr, cmap='coolwarm', figsize=(12,12))
#     plt.title("Clustermap: Feature Correlation", pad=100)
#     cg.savefig("graphs/output/breast_clustermap_corr.png")
#     plt.close()

#     # 10. Pairplot colored by diagnosis (few selected features for speed)
#     pairplot_df = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'diagnosis']]
#     sns.pairplot(pairplot_df, hue='diagnosis', palette=['green', 'red'])
#     plt.savefig("graphs/output/breast_pairplot_selected.png")
#     plt.close()

#     # 11. Lineplot of mean radius_mean per diagnosis category (just two points)
#     means = df.groupby('diagnosis')['radius_mean'].mean().reset_index()
#     plt.figure(figsize=(6,4))
#     sns.lineplot(x='diagnosis', y='radius_mean', data=means, marker='o')
#     plt.title("Lineplot: Mean Radius Mean per Diagnosis")
#     plt.xticks([0,1], ['Benign', 'Malignant'])
#     plt.savefig("graphs/output/breast_lineplot_mean_radius.png")
#     plt.close()

#     print("Breast cancer graphs generated and saved in graphs/output/")

# def generate_all_lung_cancer_graphs():
#     print("\n=== Lung Cancer Dataset Graphs ===")
#     df = pd.read_csv("../data/lung_cancer.csv")
#     df.columns = [col.strip() for col in df.columns]

#     # Mapping categories to binary numeric
#     mapping_gender = {"M":1, "F":0}
#     mapping_yes_no = {"No":0, "Yes":1}
#     df["GENDER"] = df["GENDER"].map(mapping_gender)
    
#     yes_no_cols = ["SMOKING","YELLOW_FINGERS","ANXIETY","PEER_PRESSURE","CHRONIC DISEASE",
#                    "FATIGUE","ALLERGY","WHEEZING","ALCOHOL CONSUMING","COUGHING",
#                    "SHORTNESS OF BREATH","SWALLOWING DIFFICULTY","CHEST PAIN","LUNG_CANCER"]
#     for col in yes_no_cols:
#         df[col] = df[col].map(mapping_yes_no)

#     # 1. Countplot Lung Cancer diagnosis
#     plt.figure(figsize=(6,4))
#     sns.countplot(x='LUNG_CANCER', data=df)
#     plt.title("Countplot: Lung Cancer Diagnosis (0=No,1=Yes)")
#     plt.savefig("graphs/output/lung_countplot_cancer.png")
#     plt.close()

#     # 2. Pie chart Lung Cancer diagnosis distribution
#     plt.figure(figsize=(6,6))
#     df['LUNG_CANCER'].value_counts().plot.pie(autopct='%1.1f%%', labels=['No', 'Yes'], colors=['#66b3ff','#ff6666'])
#     plt.title("Pie Chart: Lung Cancer Distribution")
#     plt.ylabel('')
#     plt.savefig("graphs/output/lung_piechart_cancer.png")
#     plt.close()

#     # 3. Histogram of AGE
#     plt.figure(figsize=(6,4))
#     plt.hist(df['AGE'], bins=30, color='skyblue')
#     plt.title("Histogram: Age Distribution")
#     plt.xlabel("Age")
#     plt.ylabel("Frequency")
#     plt.savefig("graphs/output/lung_histogram_age.png")
#     plt.close()

#     # 4. Distplot of AGE with KDE
#     plt.figure(figsize=(6,4))
#     sns.histplot(df['AGE'], kde=True, color='orange')
#     plt.title("Distplot: Age")
#     plt.xlabel("Age")
#     plt.savefig("graphs/output/lung_distplot_age.png")
#     plt.close()

#     # 5. Boxplot Age by Lung Cancer diagnosis
#     plt.figure(figsize=(6,4))
#     sns.boxplot(x='LUNG_CANCER', y='AGE', data=df, palette='Set2')
#     plt.title("Boxplot: Age by Lung Cancer Diagnosis")
#     plt.savefig("graphs/output/lung_boxplot_age.png")
#     plt.close()

#     # 6. Scatterplot AGE vs GENDER colored by Lung Cancer
#     plt.figure(figsize=(6,4))
#     sns.scatterplot(x='AGE', y='GENDER', hue='LUNG_CANCER', data=df, palette=['green','red'])
#     plt.title("Scatterplot: Age vs Gender by Lung Cancer")
#     plt.savefig("graphs/output/lung_scatterplot_age_gender.png")
#     plt.close()

#     # 7. Barplot of mean AGE by Lung Cancer diagnosis
#     plt.figure(figsize=(6,4))
#     sns.barplot(x='LUNG_CANCER', y='AGE', data=df, palette='pastel')
#     plt.title("Barplot: Mean Age by Lung Cancer Diagnosis")
#     plt.savefig("graphs/output/lung_barplot_mean_age.png")
#     plt.close()

#     # 8. Heatmap of correlation matrix
#     plt.figure(figsize=(12,10))
#     corr = df.corr()
#     sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f", cbar=True)
#     plt.title("Heatmap: Feature Correlation")
#     plt.savefig("graphs/output/lung_heatmap_corr.png")
#     plt.close()

#     # 9. Clustermap of correlation matrix
#     cg = sns.clustermap(corr, cmap='coolwarm', figsize=(12,12))
#     plt.title("Clustermap: Feature Correlation", pad=100)
#     cg.savefig("graphs/output/lung_clustermap_corr.png")
#     plt.close()

#     # 10. Pairplot of selected features by Lung Cancer
#     pairplot_cols = ['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'LUNG_CANCER']
#     sns.pairplot(df[pairplot_cols], hue='LUNG_CANCER', palette=['green','red'])
#     plt.savefig("graphs/output/lung_pairplot_selected.png")
#     plt.close()

#     # 11. Lineplot: Mean AGE per Lung Cancer diagnosis
#     means = df.groupby('LUNG_CANCER')['AGE'].mean().reset_index()
#     plt.figure(figsize=(6,4))
#     sns.lineplot(x='LUNG_CANCER', y='AGE', data=means, marker='o')
#     plt.title("Lineplot: Mean Age per Lung Cancer Diagnosis")
#     plt.xticks([0,1], ['No', 'Yes'])
#     plt.savefig("graphs/output/lung_lineplot_mean_age.png")
#     plt.close()

#     print("Lung cancer graphs generated and saved in graphs/output/")

# if __name__ == "__main__":
#     print("Generating all graphs for Breast Cancer and Lung Cancer datasets...\n")
#     generate_all_breast_cancer_graphs()
#     generate_all_lung_cancer_graphs()
#     print("\nAll graphs saved in 'graphs/output/' folder.")

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
os.makedirs("graphs/output", exist_ok=True)

def generate_all_breast_cancer_graphs():
    print("\n=== Breast Cancer Dataset Graphs & What They’re Telling You ===")
    df = pd.read_csv("../data/breast_cancer.csv")
    df = df.drop(columns=["Unnamed: 32", "id"])
    df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})

    # 1. Countplot diagnosis
    # "Hey! I’m showing you how many benign and malignant cases you have.
    # See? More benign cases here."
    print("Diagnosis value counts:")
    print(df['diagnosis'].value_counts())
    plt.figure(figsize=(6,4))
    sns.countplot(x='diagnosis', data=df)
    plt.title("Countplot: Diagnosis (0=Benign, 1=Malignant)")
    plt.savefig("graphs/output/breast_countplot_diagnosis.png")
    plt.close()

    # 2. Pie chart diagnosis
    # "Look at me! I’m the pie that shows what part of all cases are benign or malignant.
    # The bigger slice means more benign cases."
    print("\nPie chart distribution (percentage) of diagnosis:")
    print(df['diagnosis'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
    plt.figure(figsize=(6,6))
    df['diagnosis'].value_counts().plot.pie(
        autopct='%1.1f%%', labels=['Benign', 'Malignant'], colors=['#66b3ff','#ff6666']
    )
    plt.title("Pie Chart: Diagnosis Distribution")
    plt.ylabel('')
    plt.savefig("graphs/output/breast_piechart_diagnosis.png")
    plt.close()

    # 3. Histogram radius_mean
    # "I’m showing you how common different radius_mean values are.
    # Most tumors have radius_mean around here."
    print("\nRadius Mean - Summary statistics:")
    print(df['radius_mean'].describe())
    plt.figure(figsize=(6,4))
    plt.hist(df['radius_mean'], bins=30, color='skyblue')
    plt.title("Histogram: Radius Mean")
    plt.xlabel("Radius Mean")
    plt.ylabel("Frequency")
    plt.savefig("graphs/output/breast_histogram_radius_mean.png")
    plt.close()

    # 4. Distplot texture_mean
    # "Here’s a smooth curve of texture_mean.
    # It tells you where values are packed tightly or spread out."
    print("\nTexture Mean - Summary statistics:")
    print(df['texture_mean'].describe())
    plt.figure(figsize=(6,4))
    sns.histplot(df['texture_mean'], kde=True, color='orange')
    plt.title("Distplot: Texture Mean")
    plt.xlabel("Texture Mean")
    plt.savefig("graphs/output/breast_distplot_texture_mean.png")
    plt.close()

    # 5. Boxplot perimeter_mean by diagnosis
    # "Look at me! I’m a boxplot showing the spread of perimeter_mean for benign and malignant.
    # The middle line is the median. If my boxes are apart, these groups are different."
    print("\nBoxplot: Perimeter Mean stats by diagnosis:")
    print(df.groupby('diagnosis')['perimeter_mean'].describe())
    plt.figure(figsize=(6,4))
    sns.boxplot(x='diagnosis', y='perimeter_mean', data=df, palette='Set2')
    plt.title("Boxplot: Perimeter Mean by Diagnosis")
    plt.savefig("graphs/output/breast_boxplot_perimeter_mean.png")
    plt.close()

    # 6. Scatterplot radius_mean vs texture_mean by diagnosis
    # "I’m a scatterplot showing radius_mean vs texture_mean.
    # If red and green dots stay apart, I’m telling you these features help separate classes."
    print("\nScatterplot: radius_mean vs texture_mean samples by diagnosis:")
    print("Malignant samples:")
    print(df[df['diagnosis']==1][['radius_mean','texture_mean']].head())
    print("Benign samples:")
    print(df[df['diagnosis']==0][['radius_mean','texture_mean']].head())
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='radius_mean', y='texture_mean', hue='diagnosis', data=df, palette=['green','red'])
    plt.title("Scatterplot: Radius Mean vs Texture Mean")
    plt.savefig("graphs/output/breast_scatterplot_radius_texture.png")
    plt.close()

    # 7. Barplot mean area_mean by diagnosis
    # "I’m a barplot showing average tumor size (area_mean) in each group.
    # See? Malignant tumors tend to be bigger on average."
    print("\nMean Area by Diagnosis:")
    print(df.groupby('diagnosis')['area_mean'].mean())
    plt.figure(figsize=(6,4))
    sns.barplot(x='diagnosis', y='area_mean', data=df, palette='pastel')
    plt.title("Barplot: Mean Area by Diagnosis")
    plt.savefig("graphs/output/breast_barplot_area_mean.png")
    plt.close()

    # 8. Heatmap correlation matrix
    # "I’m a heatmap showing how features relate to each other.
    # Dark colors mean strong connection; light colors mean weak or no connection."
    corr = df.corr()
    print("\nHeatmap correlation matrix snippet:")
    print(corr.iloc[:5,:5])
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title("Heatmap: Feature Correlation")
    plt.savefig("graphs/output/breast_heatmap_corr.png")
    plt.close()

    # 9. Clustermap correlation matrix
    # "I group features that behave similarly, so you can see clusters of related features."
    plt.figure(figsize=(12,12))
    cg = sns.clustermap(corr, cmap='coolwarm')
    plt.title("Clustermap: Feature Correlation", pad=100)
    cg.savefig("graphs/output/breast_clustermap_corr.png")
    plt.close()

    # 10. Pairplot selected features
    # "Here, I show all pairs of important features to spot patterns and differences by diagnosis."
    pairplot_df = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'diagnosis']]
    print("\nPairplot shows how features pair up and differ between benign and malignant.")
    sns.pairplot(pairplot_df, hue='diagnosis', palette=['green', 'red'])
    plt.savefig("graphs/output/breast_pairplot_selected.png")
    plt.close()

    # 11. Lineplot mean radius_mean per diagnosis
    # "I’m a lineplot showing average radius_mean for benign and malignant tumors.
    # You see the malignant line is higher? That means bigger radius on average."
    means = df.groupby('diagnosis')['radius_mean'].mean().reset_index()
    print("\nMean radius_mean per diagnosis for lineplot:")
    print(means)
    plt.figure(figsize=(6,4))
    sns.lineplot(x='diagnosis', y='radius_mean', data=means, marker='o')
    plt.title("Lineplot: Mean Radius Mean per Diagnosis")
    plt.xticks([0,1], ['Benign', 'Malignant'])
    plt.savefig("graphs/output/breast_lineplot_mean_radius.png")
    plt.close()


def generate_all_lung_cancer_graphs():
    print("\n=== Lung Cancer Dataset Graphs & What They’re Telling You ===")
    df = pd.read_csv("../data/lung_cancer.csv")
    df.columns = [col.strip() for col in df.columns]

    mapping_gender = {"M":1, "F":0}
    mapping_yes_no = {"No":0, "Yes":1}
    df["GENDER"] = df["GENDER"].map(mapping_gender)
    
    yes_no_cols = ["SMOKING","YELLOW_FINGERS","ANXIETY","PEER_PRESSURE","CHRONIC DISEASE",
                   "FATIGUE","ALLERGY","WHEEZING","ALCOHOL CONSUMING","COUGHING",
                   "SHORTNESS OF BREATH","SWALLOWING DIFFICULTY","CHEST PAIN","LUNG_CANCER"]
    for col in yes_no_cols:
        df[col] = df[col].map(mapping_yes_no)

    # 1. Countplot Lung Cancer diagnosis
    # "Hey! This shows how many have lung cancer and how many don’t.
    # More no’s here."
    print("Lung Cancer diagnosis value counts:")
    print(df['LUNG_CANCER'].value_counts())
    plt.figure(figsize=(6,4))
    sns.countplot(x='LUNG_CANCER', data=df)
    plt.title("Countplot: Lung Cancer Diagnosis (0=No,1=Yes)")
    plt.savefig("graphs/output/lung_countplot_cancer.png")
    plt.close()

    # 2. Pie chart Lung Cancer diagnosis
    # "This pie shows what part of the group has lung cancer.
    # See the big slice? Those without lung cancer."
    print("\nLung Cancer diagnosis distribution (percentage):")
    print(df['LUNG_CANCER'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
    plt.figure(figsize=(6,6))
    df['LUNG_CANCER'].value_counts().plot.pie(
        autopct='%1.1f%%', labels=['No', 'Yes'], colors=['#66b3ff','#ff6666']
    )
    plt.title("Pie Chart: Lung Cancer Distribution")
    plt.ylabel('')
    plt.savefig("graphs/output/lung_piechart_cancer.png")
    plt.close()

    # 3. Histogram AGE
    # "How old are the patients? This histogram shows age counts.
    # Most patients fall here."
    print("\nAge distribution summary:")
    print(df['AGE'].describe())
    plt.figure(figsize=(6,4))
    plt.hist(df['AGE'], bins=30, color='skyblue')
    plt.title("Histogram: Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.savefig("graphs/output/lung_histogram_age.png")
    plt.close()

    # 4. Distplot AGE
    # "Here’s a smooth curve showing age distribution.
    # Peaks mean more patients at that age."
    plt.figure(figsize=(6,4))
    sns.histplot(df['AGE'], kde=True, color='orange')
    plt.title("Distplot: Age")
    plt.xlabel("Age")
    plt.savefig("graphs/output/lung_distplot_age.png")
    plt.close()

    # 5. Boxplot Age by Lung Cancer
    # "Compare age spread for people with and without lung cancer.
    # If boxes look different, age may matter."
    print("\nBoxplot stats: Age by Lung Cancer diagnosis:")
    print(df.groupby('LUNG_CANCER')['AGE'].describe())
    plt.figure(figsize=(6,4))
    sns.boxplot(x='LUNG_CANCER', y='AGE', data=df, palette='Set2')
    plt.title("Boxplot: Age by Lung Cancer Diagnosis")
    plt.savefig("graphs/output/lung_boxplot_age.png")
    plt.close()

    # 6. Scatterplot Age vs Gender by Lung Cancer
    # "Age vs gender colored by lung cancer. 
    # Spot if men or women in certain ages have more lung cancer."
    print("\nScatterplot examples Age vs Gender by Lung Cancer:")
    print("Sample Lung Cancer=Yes:")
    print(df[df['LUNG_CANCER']==1][['AGE','GENDER']].head())
    print("Sample Lung Cancer=No:")
    print(df[df['LUNG_CANCER']==0][['AGE','GENDER']].head())
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='AGE', y='GENDER', hue='LUNG_CANCER', data=df, palette=['green','red'])
    plt.title("Scatterplot: Age vs Gender by Lung Cancer")
    plt.savefig("graphs/output/lung_scatterplot_age_gender.png")
    plt.close()

    # 7. Barplot Smoking by Lung Cancer
    # "See how smoking relates to lung cancer in bars.
    # Higher bars in smokers with cancer means risk is real."
    print("\nMean Smoking rate by Lung Cancer diagnosis:")
    print(df.groupby('LUNG_CANCER')['SMOKING'].mean())
    plt.figure(figsize=(6,4))
    sns.barplot(x='LUNG_CANCER', y='SMOKING', data=df, palette='pastel')
    plt.title("Barplot: Smoking Rate by Lung Cancer")
    plt.savefig("graphs/output/lung_barplot_smoking.png")
    plt.close()

    # 8. Heatmap correlation matrix
    # "I’m showing you how lung cancer features relate to each other.
    # Darker color means stronger connection."
    corr = df.corr()
    print("\nHeatmap correlation matrix snippet:")
    print(corr.iloc[:5,:5])
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title("Heatmap: Lung Cancer Feature Correlation")
    plt.savefig("graphs/output/lung_heatmap_corr.png")
    plt.close()

    # 9. Clustermap correlation matrix
    # "I group related features, so you see clusters that behave similarly."
    plt.figure(figsize=(12,12))
    cg = sns.clustermap(corr, cmap='coolwarm')
    plt.title("Clustermap: Lung Cancer Feature Correlation", pad=100)
    cg.savefig("graphs/output/lung_clustermap_corr.png")
    plt.close()

    # 10. Pairplot some features by Lung Cancer
    # "I’m a pairplot showing how features pair up and differ between those with and without lung cancer."
    pairplot_df = df[['AGE','SMOKING','YELLOW_FINGERS','ANXIETY','LUNG_CANCER']]
    print("\nPairplot info:")
    print(pairplot_df.head())
    sns.pairplot(pairplot_df, hue='LUNG_CANCER', palette=['green', 'red'])
    plt.savefig("graphs/output/lung_pairplot_selected.png")
    plt.close()

    # 11. Lineplot Mean Age by Lung Cancer
    # "I’m a lineplot showing average age for people with and without lung cancer.
    # See the difference? It’s interesting!"
    means = df.groupby('LUNG_CANCER')['AGE'].mean().reset_index()
    print("\nMean Age per Lung Cancer diagnosis for lineplot:")
    print(means)
    plt.figure(figsize=(6,4))
    sns.lineplot(x='LUNG_CANCER', y='AGE', data=means, marker='o')
    plt.title("Lineplot: Mean Age per Lung Cancer Diagnosis")
    plt.xticks([0,1], ['No', 'Yes'])
    plt.savefig("graphs/output/lung_lineplot_mean_age.png")
    plt.close()

if __name__ == "__main__":
    print("Generating all graphs with simple explanations...\n")
    generate_all_breast_cancer_graphs()
    generate_all_lung_cancer_graphs()
    print("\nAll graphs generated and saved in 'graphs/output/'. Check your terminal for what the graphs are telling you!")
