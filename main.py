import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np
from datasets import load_dataset
from scipy import stats

from utils.settings import Settings, get_settings

settings: Settings = get_settings()


def main():  
    # github data
    github_data = load_dataset(
        "ibragim-bad/github-repos-metadata-40M", split=settings.github_split
    )
    github_data_df = github_data.to_pandas()
    github_data_df = clean_dataset(github_data_df)

    # filter ' ' languages to be N/A, this is usually due 
    # to a repo having multiple languages
    github_data_df["language"] = (
        github_data_df["language"]
        .astype(object)
        .where(github_data_df["language"].notna(), pd.NA)
    )
    github_data_df["language"] = github_data_df["language"].apply(
        lambda s: s.strip() if isinstance(s, str) else s
    )
    github_data_df["language"].replace("", pd.NA, inplace=True)

    github_data_df.dropna(
        subset=["language"], inplace=True
    )  # we need only repos with code with a primary programming language

    # stack overflow data time

    stack_overflow_df = pd.read_csv(settings.survey_csv)
    stack_overflow_df = clean_dataset(stack_overflow_df)

    # make sure age/years of exp is not negative
    survey_cols = ["Age", "WorkExp", "YearsCode"]

    for col in survey_cols:
        stack_overflow_df[col] = pd.to_numeric(stack_overflow_df[col], errors="coerce")

        neg_mask = stack_overflow_df[col] < 0
        if neg_mask.any():
            stack_overflow_df.loc[neg_mask, col] = pd.NA

    stack_overflow_df.dropna(inplace=True)

    gh_lang_freq = github_data_df["language"].value_counts() 

    plot_gh_freq(gh_lang_freq)
    
    # Extract Stack Overflow language preferences
    lang_col = None
    for possible_col in ['LanguageHaveWorkedWith', 'LanguageWorkedWith', 'LanguageWantToWorkWith']:
        if possible_col in stack_overflow_df.columns:
            lang_col = possible_col
            break
    
    if lang_col is None:
        print("Warning: Could not find language column in Stack Overflow survey")
        print(f"Available columns: {stack_overflow_df.columns.tolist()}")
        return
    
    # Split semicolon-separated languages and count
    so_languages = (
        stack_overflow_df[lang_col]
        .dropna()
        .str.split(';')
        .explode()
        .str.strip()
    )
    so_lang_freq = so_languages.value_counts()
    
    print(f"\nStack Overflow Language Preferences (from {lang_col}):")
    print(so_lang_freq.head(20))
    
    # Merge the two datasets on common languages
    merged = pd.DataFrame({
        'github_count': gh_lang_freq,
        'stackoverflow_count': so_lang_freq
    }).dropna()
    
    if len(merged) == 0:
        print("No common languages found between datasets")
        return
    
    print(f"\nFound {len(merged)} common languages")
    print("\nTop 10 languages by GitHub count:")
    print(merged.nlargest(10, 'github_count'))
    
    # Perform linear regression with statistical tests
    perform_regression_analysis(merged)


def perform_regression_analysis(merged_df):
    """Perform linear regression and calculate t-test and statistical measures."""
    X = merged_df['stackoverflow_count'].values
    y = merged_df['github_count'].values
    n = len(X)
    
    # Calculate regression statistics using scipy
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
    
    # Calculate predictions and residuals
    y_pred = slope * X + intercept
    residuals = y - y_pred
    
    # Calculate additional statistics
    ss_res = np.sum(residuals**2)  # Sum of squared residuals
    ss_tot = np.sum((y - np.mean(y))**2)  # Total sum of squares
    r_squared = r_value**2
    
    # Degrees of freedom
    df_residual = n - 2
    
    # Standard error of the estimate
    se_estimate = np.sqrt(ss_res / df_residual)
    
    # Standard error of the slope
    se_slope = std_err
    
    # t-statistic for slope
    t_stat = slope / se_slope
    
    # Confidence interval for slope (95%)
    t_critical = stats.t.ppf(0.975, df_residual)
    ci_lower = slope - t_critical * se_slope
    ci_upper = slope + t_critical * se_slope
    
    # F-statistic
    f_stat = (r_squared / 1) / ((1 - r_squared) / df_residual)
    f_p_value = 1 - stats.f.cdf(f_stat, 1, df_residual)
    
    # Print results
    print("\n" + "="*60)
    print("LINEAR REGRESSION ANALYSIS")
    print("="*60)
    print(f"\nRegression Equation: y = {slope:.4f}x + {intercept:.4f}")
    print(f"\nSample Size: n = {n}")
    print(f"\nCoefficient of Determination:")
    print(f"  R² = {r_squared:.4f}")
    print(f"  R = {r_value:.4f}")
    print(f"\nRegression Coefficients:")
    print(f"  Slope (β₁) = {slope:.4f}")
    print(f"  Intercept (β₀) = {intercept:.4f}")
    print(f"\nStandard Errors:")
    print(f"  SE(slope) = {se_slope:.4f}")
    print(f"  SE(estimate) = {se_estimate:.4f}")
    print(f"\nHypothesis Test for Slope:")
    print(f"  H₀: β₁ = 0 (no relationship)")
    print(f"  H₁: β₁ ≠ 0 (relationship exists)")
    print(f"  t-statistic = {t_stat:.4f}")
    print(f"  p-value = {p_value:.4e}")
    print(f"  Degrees of Freedom = {df_residual}")
    
    if p_value < 0.001:
        print(f"  Result: Reject H₀ (p < 0.001) ***")
    elif p_value < 0.01:
        print(f"  Result: Reject H₀ (p < 0.01) **")
    elif p_value < 0.05:
        print(f"  Result: Reject H₀ (p < 0.05) *")
    else:
        print(f"  Result: Fail to reject H₀ (p ≥ 0.05)")
    
    print(f"\n95% Confidence Interval for Slope:")
    print(f"  [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"\nF-statistic (ANOVA):")
    print(f"  F = {f_stat:.4f}")
    print(f"  p-value = {f_p_value:.4e}")
    print(f"\nResidual Analysis:")
    print(f"  Sum of Squared Residuals (SSR) = {ss_res:.2f}")
    print(f"  Total Sum of Squares (SST) = {ss_tot:.2f}")
    print("="*60 + "\n")
    
    # Plot regression
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot with regression line
    ax1.scatter(X, y, alpha=0.6, s=50)
    ax1.plot(X, y_pred, 'r-', linewidth=2, 
             label=f'y = {slope:.2f}x + {intercept:.2f}\nR² = {r_squared:.4f}')
    
    # Label top languages
    top_langs = merged_df.nlargest(10, 'github_count')
    for lang, row in top_langs.iterrows():
        ax1.annotate(lang, (row['stackoverflow_count'], row['github_count']), 
                    fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    
    ax1.set_xlabel('Stack Overflow Developer Preference Count')
    ax1.set_ylabel('GitHub Repository Count')
    ax1.set_title(f'GitHub Repos vs Stack Overflow Preferences\np-value = {p_value:.4e}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    ax2.scatter(y_pred, residuals, alpha=0.6, s=50)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Fitted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('github_vs_stackoverflow_regression.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_gh_freq(gh_lang_freq):
     # Plot top 20 languages
    top_n = 20
    gh_lang_freq.head(top_n).plot(kind='barh', figsize=(10, 8))
    plt.title(f'Top {top_n} Programming Languages on GitHub')
    plt.xlabel('Number of Repositories')
    plt.ylabel('Language')
    plt.tight_layout()
    plt.gca().invert_yaxis()  # Most frequent at top
    plt.savefig('github_language_frequency.png', dpi=300, bbox_inches='tight')
    plt.show()

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Preforms common data cleaning tasks."""
    ret = df.copy()

    ret.drop_duplicates()

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in num_cols:
        ret[col].fillna(0, inplace=True)

    return ret


if __name__ == "__main__":
    main()
