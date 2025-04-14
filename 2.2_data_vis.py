############################################################################
"""
This file Just has extra visualizations for the presentation and report
"""
############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import umap

sns.set(style="whitegrid")


def main():

    data_path = "CSV files/final_cleaned_file.csv"  # Make sure this CSV is your cleaned data file
    df = pd.read_csv(data_path)

    print("DataFrame shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("First 5 rows:")
    print(df.head())

    claims = df["claim"].dropna().tolist()
    print(f"Total claims available for visualization: {len(claims)}")

    all_claims_text = " ".join(claims)

    wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        max_words=200,
        contour_width=1,
        contour_color="steelblue"
    ).generate(all_claims_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Claims")
    plt.tight_layout()
    plt.savefig("Images/word_cloud_claims.png", dpi=300)
    plt.show()
    print("Word Cloud saved as 'word_cloud_claims.png'")

    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = tfidf_vectorizer.fit_transform(claims)
    print("TF-IDF feature matrix shape:", X_tfidf.shape)

    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    X_umap = umap_reducer.fit_transform(X_tfidf)
    print("UMAP reduced feature shape:", X_umap.shape)

    if "label" in df.columns:
        labels = df["label"].dropna().values[:len(X_umap)]
    else:
        labels = None

    plt.figure(figsize=(12, 8))
    if labels is not None:
        scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap="viridis", s=10, alpha=0.7)
        plt.colorbar(scatter, label="Label")
    else:
        plt.scatter(X_umap[:, 0], X_umap[:, 1], s=10, alpha=0.7)

    plt.title("UMAP Projection of TF-IDF Features")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.tight_layout()
    plt.savefig("Images/umap_projection.png", dpi=300)
    plt.show()
    print("UMAP projection saved as 'umap_projection.png'")

    claim_lengths = [len(claim.split()) for claim in claims]

    plt.figure(figsize=(10, 6))
    sns.histplot(claim_lengths, bins=30, kde=True, color="teal")
    plt.title("Distribution of Claim Lengths")
    plt.xlabel("Number of Words per Claim")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("Images/claim_length_distribution.png", dpi=300)
    plt.show()
    print("Claim length distribution histogram saved as 'claim_length_distribution.png'")


if __name__ == '__main__':
    main()


############################################################################
# This is the result that I got after running this file
"""
DataFrame shape: (8897, 6)
Columns: ['claim_id', 'claim', 'main_text', 'label', 'claim-p', 'postagged']
First 5 rows:
   claim_id  ...                                          postagged
0     10166  ...  [('study', 'NN'), ('vaccine', 'NN'), ('for', '...
1      9851  ...  [('angioplasty', 'NN'), ('through', 'IN'), ('t...
2      2768  ...  [('u', 'JJ'), ('s', 'NN'), ('says', 'VBZ'), ('...
3     28215  ...  [('opossums', 'NNS'), ('kill', 'VB'), ('thousa...
4      5793  ...  [('democrats', 'NNS'), ('hoping', 'VBG'), ('to...

[5 rows x 6 columns]
Total claims available for visualization: 8897
Word Cloud saved as 'word_cloud_claims.png'
TF-IDF feature matrix shape: (8897, 5000)
/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/sklearn/utils/deprecation.py:151: FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.
  warnings.warn(
/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
UMAP reduced feature shape: (8897, 2)
UMAP projection saved as 'umap_projection.png'
Claim length distribution histogram saved as 'claim_length_distribution.png'

Process finished with exit code 0
"""