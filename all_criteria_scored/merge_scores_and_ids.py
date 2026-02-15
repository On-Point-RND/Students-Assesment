import pandas as pd

# Define paths
csv_files = ["/home/dev/work_main/random/assesment/all_criteria_scored/mistral_nemo/cv_scores_mistralai_mistral-nemo.csv",
             "/home/dev/work_main/random/assesment/all_criteria_scored/mistral_nemo/motivation_scores_mistralai_mistral-nemo.csv",
             "/home/dev/work_main/random/assesment/all_criteria_scored/mistral_nemo/presentation_scores_mistralai_mistral-nemo.csv"]
parquet_path = "./data/dataset.parquet"


def clean_name(name):
    """Improved cleaning for document names."""
    if pd.isna(name):
        return name
    
    name = str(name)
    
    # Remove .txt extension
    name = name.replace('.txt', '')
    
    # Remove hyphens
    name = name.replace('-', '')
    
    # Decode Unicode escape sequences like #U0161
    def decode_unicode(match):
        hex_code = match.group(1)
        try:
            return chr(int(hex_code, 16))
        except:
            return match.group(0)
    
    name = re.sub(r'#U([0-9a-fA-F]{4})', decode_unicode, name)
    
    # Remove trailing dots and spaces
    name = name.strip('. ')
    
    return name

def is_valid_document_name(name):
    """Check if name looks like a real document name (not a metadata file)."""
    if pd.isna(name):
        return False
    
    name = str(name).lower()
    
    # Exclude obvious file names/metadata
    exclude_patterns = [
        'prof_eval_anon',
        'anonymized_test_project_scores_final',
        'offline_tests',
        'anon_projects_scores',
        '.xls',
        '.cs',
        '.csv',
        'test_',
        'eval_',
        'anon_',
        'score',
        'project_scores'
    ]
    
    for pattern in exclude_patterns:
        if pattern in name:
            return False
    
    # Should contain at least one letter and look like a person's name
    if not re.search(r'[a-z]', name):
        return False
    
    return True


# Read parquet and select specific columns
scores = pd.read_parquet(parquet_path)[['school_participation_flag', 'project_participation_flag']]

# Clean score indices
scores.index = scores.index.map(clean_name)

# Filter out non-document entries from scores
valid_mask = scores.index.map(is_valid_document_name)
scores_filtered = scores[valid_mask].copy()

print(f"Original scores: {len(scores)} entries")
print(f"Filtered scores: {len(scores_filtered)} entries")
print(f"Removed {len(scores) - len(scores_filtered)} non-document entries")

# Show what was removed
removed = scores[~valid_mask]
if len(removed) > 0:
    print("\nRemoved entries (first 10):")
    for name in removed.index[:10]:
        print(f"  - {name}")

# Process each CSV
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    
    # Clean document names
    df['cleaned'] = df['document_name'].apply(clean_name)
    
    # Get intersection with filtered scores
    csv_names = set(df['cleaned'])
    score_indices = set(scores_filtered.index)
    intersection = csv_names.intersection(score_indices)
    
    # Print intersection info
    print(f"\n{csv_file}:")
    print(f"  CSV names: {len(csv_names)}")
    print(f"  Valid score indices: {len(score_indices)}")
    print(f"  Intersection: {len(intersection)}")
    print(f"  Match rate: {len(intersection)/len(csv_names)*100:.1f}%")
    
    # Print non-intersecting names
    non_intersecting_csv = csv_names - score_indices
    non_intersecting_scores = score_indices - csv_names
    
    if non_intersecting_csv:
        print(f"  Names in CSV but not in scores ({len(non_intersecting_csv)}):")
        for name in list(non_intersecting_csv)[:5]:
            print(f"    - {name}")
        if len(non_intersecting_csv) > 5:
            print(f"    ... and {len(non_intersecting_csv) - 5} more")
    
    if non_intersecting_scores:
        print(f"  Names in scores but not in CSV ({len(non_intersecting_scores)}):")
        for name in list(non_intersecting_scores)[:5]:
            print(f"    - {name}")
        if len(non_intersecting_scores) > 5:
            print(f"    ... and {len(non_intersecting_scores) - 5} more")
    
    # Merge with filtered scores
    merged = df.merge(scores_filtered, left_on='cleaned', right_index=True, how='left')
    merged = merged.drop('cleaned', axis=1)
    
    # Save
    merged.to_csv(csv_file.replace('.csv', '_merged.csv'), index=False)
    print(f"  Saved: {csv_file.replace('.csv', '_merged.csv')}")