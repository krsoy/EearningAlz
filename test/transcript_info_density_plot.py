import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

OUT = Path("transcript_information_density")
PLOT_DIR = OUT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

density_path = OUT / "transcript_information_density.csv"
term_freq_path = OUT / "supply_chain_term_global_frequency.csv"
company_freq_path = OUT / "company_entity_global_frequency.csv"
company_density_path = OUT / "company_level_information_density.csv"

density_df = pd.read_csv(density_path)
term_freq_df = pd.read_csv(term_freq_path)
company_freq_df = pd.read_csv(company_freq_path)
company_density_df = pd.read_csv(company_density_path)

# ============================================================
# 1. Distribution of supply-chain terms per transcript
# ============================================================

plt.figure(figsize=(10, 6))
plt.hist(
    density_df["supply_chain_terms_per_1000_words"].dropna(),
    bins=40
)
plt.title("Distribution of Supply Chain Terms per 1,000 Words")
plt.xlabel("Supply Chain Terms per 1,000 Words")
plt.ylabel("Number of Transcripts")
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_supply_chain_terms_distribution.png", dpi=300)
plt.show()

# ============================================================
# 2. Distribution of company mentions per transcript
# ============================================================

plt.figure(figsize=(10, 6))
plt.hist(
    density_df["company_mentions_per_1000_words"].dropna(),
    bins=40
)
plt.title("Distribution of Company Mentions per 1,000 Words")
plt.xlabel("Company Mentions per 1,000 Words")
plt.ylabel("Number of Transcripts")
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_company_mentions_distribution.png", dpi=300)
plt.show()

# ============================================================
# 3. Supply-chain information density vs company entity density
# ============================================================

plt.figure(figsize=(10, 7))
plt.scatter(
    density_df["supply_chain_terms_per_1000_words"],
    density_df["company_mentions_per_1000_words"],
    alpha=0.45
)
plt.title("Transcript Information Density")
plt.xlabel("Supply Chain Terms per 1,000 Words")
plt.ylabel("Company Mentions per 1,000 Words")
plt.tight_layout()
plt.savefig(PLOT_DIR / "03_supply_chain_vs_company_mentions.png", dpi=300)
plt.show()

# ============================================================
# 4. Top transcripts by information density score
# ============================================================

top_n = 20
top_transcripts = density_df.sort_values(
    "information_density_score",
    ascending=False
).head(top_n).copy()

top_transcripts["label"] = (
    top_transcripts["ticker"].astype(str)
    + " | "
    + top_transcripts["company"].astype(str).str.slice(0, 35)
)

plt.figure(figsize=(12, 8))
plt.barh(
    top_transcripts["label"][::-1],
    top_transcripts["information_density_score"][::-1]
)
plt.title(f"Top {top_n} Transcripts by Information Density Score")
plt.xlabel("Information Density Score")
plt.ylabel("Transcript")
plt.tight_layout()
plt.savefig(PLOT_DIR / "04_top_transcripts_information_density.png", dpi=300)
plt.show()

# ============================================================
# 5. Top supply-chain terms by total occurrences
# ============================================================

top_terms = term_freq_df.sort_values(
    "total_occurrences",
    ascending=False
).head(30)

plt.figure(figsize=(12, 9))
plt.barh(
    top_terms["term"][::-1],
    top_terms["total_occurrences"][::-1]
)
plt.title("Top Supply Chain Terms by Total Occurrences")
plt.xlabel("Total Occurrences")
plt.ylabel("Supply Chain Term")
plt.tight_layout()
plt.savefig(PLOT_DIR / "05_top_supply_chain_terms.png", dpi=300)
plt.show()

# ============================================================
# 6. Top supply-chain terms by transcript coverage
# ============================================================

top_coverage_terms = term_freq_df.sort_values(
    "transcript_coverage_pct",
    ascending=False
).head(30)

plt.figure(figsize=(12, 9))
plt.barh(
    top_coverage_terms["term"][::-1],
    top_coverage_terms["transcript_coverage_pct"][::-1]
)
plt.title("Top Supply Chain Terms by Transcript Coverage")
plt.xlabel("Transcript Coverage (%)")
plt.ylabel("Supply Chain Term")
plt.tight_layout()
plt.savefig(PLOT_DIR / "06_top_supply_chain_term_coverage.png", dpi=300)
plt.show()

# ============================================================
# 7. Top company entities by total mentions
# ============================================================

top_entities = company_freq_df.sort_values(
    "total_mentions",
    ascending=False
).head(30)

plt.figure(figsize=(12, 9))
plt.barh(
    top_entities["company_entity"][::-1],
    top_entities["total_mentions"][::-1]
)
plt.title("Top Company Entities by Total Mentions")
plt.xlabel("Total Mentions")
plt.ylabel("Company Entity")
plt.tight_layout()
plt.savefig(PLOT_DIR / "07_top_company_entities_mentions.png", dpi=300)
plt.show()

# ============================================================
# 8. Top company entities by transcript coverage
# ============================================================

top_entity_coverage = company_freq_df.sort_values(
    "transcript_coverage_pct",
    ascending=False
).head(30)

plt.figure(figsize=(12, 9))
plt.barh(
    top_entity_coverage["company_entity"][::-1],
    top_entity_coverage["transcript_coverage_pct"][::-1]
)
plt.title("Top Company Entities by Transcript Coverage")
plt.xlabel("Transcript Coverage (%)")
plt.ylabel("Company Entity")
plt.tight_layout()
plt.savefig(PLOT_DIR / "08_top_company_entity_coverage.png", dpi=300)
plt.show()

# ============================================================
# 9. Top companies by average transcript information density
# ============================================================

top_company_density = company_density_df[
    company_density_df["transcript_count"] >= 1
].sort_values(
    "avg_information_density_score",
    ascending=False
).head(30)

top_company_density["label"] = (
    top_company_density["ticker"].astype(str)
    + " | "
    + top_company_density["company"].astype(str).str.slice(0, 35)
)

plt.figure(figsize=(12, 9))
plt.barh(
    top_company_density["label"][::-1],
    top_company_density["avg_information_density_score"][::-1]
)
plt.title("Top Companies by Average Transcript Information Density")
plt.xlabel("Average Information Density Score")
plt.ylabel("Company")
plt.tight_layout()
plt.savefig(PLOT_DIR / "09_top_companies_information_density.png", dpi=300)
plt.show()

# ============================================================
# 10. Company-level supply chain density vs company mention density
# ============================================================

plt.figure(figsize=(10, 7))
plt.scatter(
    company_density_df["avg_supply_chain_terms_per_1000_words"],
    company_density_df["avg_company_mentions_per_1000_words"],
    alpha=0.55
)
plt.title("Company-Level Average Information Density")
plt.xlabel("Average Supply Chain Terms per 1,000 Words")
plt.ylabel("Average Company Mentions per 1,000 Words")
plt.tight_layout()
plt.savefig(PLOT_DIR / "10_company_level_density_scatter.png", dpi=300)
plt.show()

# ============================================================
# 11. Word count vs information density
# ============================================================

plt.figure(figsize=(10, 7))
plt.scatter(
    density_df["word_count"],
    density_df["information_density_score"],
    alpha=0.4
)
plt.title("Transcript Length vs Information Density")
plt.xlabel("Word Count")
plt.ylabel("Information Density Score")
plt.tight_layout()
plt.savefig(PLOT_DIR / "11_word_count_vs_information_density.png", dpi=300)
plt.show()

# ============================================================
# 12. Boxplot of key density metrics
# ============================================================

metrics = [
    "supply_chain_terms_per_1000_words",
    "company_mentions_per_1000_words",
    "unique_companies_per_1000_words",
    "information_density_score"
]

plt.figure(figsize=(11, 7))
plt.boxplot(
    [density_df[m].dropna() for m in metrics],
    labels=[
        "Supply Chain\nTerms",
        "Company\nMentions",
        "Unique\nCompanies",
        "Information\nDensity"
    ],
    showfliers=False
)
plt.title("Distribution of Transcript Density Metrics")
plt.ylabel("Metric Value")
plt.tight_layout()
plt.savefig(PLOT_DIR / "12_density_metrics_boxplot.png", dpi=300)
plt.show()

print(f"All plots saved to: {PLOT_DIR.resolve()}")