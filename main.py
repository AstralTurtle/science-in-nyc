import pandas as pd  # noqa: D100
from datasets import load_dataset

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

    # debug prints: check all links
    # none_per_col = github_data_df.isna().sum()
    # cols_with_none = none_per_col[none_per_col > 0]
    # if not cols_with_none.empty:
    #     print("Columns with None/NaN (count):")
    #     print(cols_with_none.to_string())
    # else:
    #     print("No columns contain None/NaN.")

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
