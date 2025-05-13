# -*- coding: utf-8 -*-
"""
STATISTICAL ANALYSIS FOR SYSTEMATIC REVIEW - OBJECTIVE 1
Examining the Impact of Interactive Digital Tools on Mathematical Skills and Computational Thinking
in Primary Education Students (6-12 years)

Author: [Your Name] - Corrected and Refactored by AI Assistant
Version: 1.1
"""

# --- 0. Import Required Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # For statistical tests
import re             # For regular expressions (text processing)
import os             # For interacting with the operating system (creating folders)
import networkx as nx # For network analysis
from wordcloud import WordCloud # For creating word clouds
import nltk           # Natural Language Toolkit for text processing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer # For TF-IDF analysis
import warnings       # To suppress warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
# Set up output directory (Changed to ResObj01a as requested)
output_dir = 'ResObj01a'
try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory '{output_dir}' created or already exists.")
except OSError as e:
    print(f"Error creating directory {output_dir}: {e}")
    # Exit if directory creation fails, as outputs cannot be saved
    exit()

# Configure visualization aesthetics
plt.style.use('seaborn-v0_8-whitegrid') # Use a visually appealing style
sns.set_palette("viridis")              # Set a consistent color palette
plt.rcParams['font.family'] = 'DejaVu Sans' # Ensure consistent font
plt.rcParams['figure.figsize'] = (12, 8)    # Set default figure size
plt.rcParams['figure.dpi'] = 100            # Set default figure resolution

# Make sure NLTK resources (punkt tokenizer, stopwords) are available
# Download them quietly if not found
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' resource not found. Downloading...")
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK 'stopwords' resource not found. Downloading...")
    nltk.download('stopwords', quiet=True)

# --- Report Initialization ---
report_lines = []
report_lines.append("=== SYSTEMATIC REVIEW ANALYSIS REPORT: OBJECTIVE 1 ===")
report_lines.append("Impact of Digital Interactive Tools on Mathematical Skills and Computational Thinking")
report_lines.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}") # Added time
report_lines.append("=" * 60)

print("Starting analysis for Objective 1...")

# --- 1. Data Loading and Preparation ---
report_lines.append("\n1. DATA LOADING AND PREPARATION")
report_lines.append("-" * 40)

data_file = 'analisisTodos.csv'
try:
    # Load the CSV file (assuming semicolon delimiter and utf-8 encoding)
    df = pd.read_csv(data_file, delimiter=';', encoding='utf-8')
    print(f"Dataset '{data_file}' loaded successfully. Shape: {df.shape}")
    report_lines.append(f"Dataset loaded with {df.shape[0]} records and {df.shape[1]} columns.")
except FileNotFoundError:
    error_msg = f"Error: Dataset file '{data_file}' not found. Please ensure it's in the same directory."
    print(error_msg)
    report_lines.append(error_msg)
    exit() # Exit if the data file is missing
except Exception as e:
    error_msg = f"Error loading the dataset: {str(e)}"
    print(error_msg)
    report_lines.append(error_msg)
    exit() # Exit on other loading errors

# Define relevant columns for Objective 1
objective1_cols = [
    'Authors', 'Year', 'Country', 'Sample Size', 'Study Design',
    'Intervention', 'Control/Comparison', 'Effect Sizes',
    'General Results',
    'Efectividad Cognitiva: Desarrollo de Habilidades Matemáticas',
    'Efectividad Cognitiva: Resolución de Problemas y Pensamiento Computacional',
    'Herramientas Digitales: Tipos de Herramientas Interactivas'
]

# Check if all required columns exist in the loaded dataframe
missing_cols = [col for col in objective1_cols if col not in df.columns]
if missing_cols:
    warning_msg = f"Warning: The following expected columns are missing: {', '.join(missing_cols)}. Analysis will proceed with available columns."
    print(warning_msg)
    report_lines.append(warning_msg)
    # Filter objective1_cols to only include columns that actually exist
    objective1_cols = [col for col in objective1_cols if col in df.columns]
    if not objective1_cols:
        error_msg = "Error: No relevant columns found for Objective 1 analysis. Exiting."
        print(error_msg)
        report_lines.append(error_msg)
        exit()

# Create the working dataframe with only the relevant (and existing) columns
df_obj1 = df[objective1_cols].copy()

# Rename columns for easier programmatic access (using snake_case)
column_mapping = {
    'Authors': 'authors',
    'Year': 'year',
    'Country': 'country',
    'Sample Size': 'sample_size',
    'Study Design': 'study_design',
    'Intervention': 'intervention',
    'Control/Comparison': 'control',
    'Effect Sizes': 'effect_sizes',
    'General Results': 'general_results',
    # Handling potentially missing columns gracefully during rename
    'Efectividad Cognitiva: Desarrollo de Habilidades Matemáticas': 'math_skills',
    'Efectividad Cognitiva: Resolución de Problemas y Pensamiento Computacional': 'problem_solving',
    'Herramientas Digitales: Tipos de Herramientas Interactivas': 'digital_tools'
}

# Apply renaming only for columns that exist in df_obj1
actual_mapping = {k: v for k, v in column_mapping.items() if k in df_obj1.columns}
df_obj1.rename(columns=actual_mapping, inplace=True)

# Create a combined cognitive text field for easier text analysis
# Fill NaN values with empty strings before concatenation
cognitive_cols_present = [col for col in ['math_skills', 'problem_solving'] if col in df_obj1.columns]
if cognitive_cols_present:
    df_obj1['cognitive_text'] = df_obj1[cognitive_cols_present].fillna('').agg(' '.join, axis=1)
else:
    df_obj1['cognitive_text'] = '' # Create empty column if source cols are missing
    report_lines.append("Warning: 'math_skills' or 'problem_solving' columns missing. 'cognitive_text' will be empty.")

# --- Data Cleaning Functions ---
def extract_sample_size(text):
    """Extracts the first number found in the sample size text."""
    if pd.isna(text):
        return np.nan
    # Find all sequences of digits
    numbers = re.findall(r'\d+', str(text))
    # Return the first number found, or NaN if none are found
    return int(numbers[0]) if numbers else np.nan

# --- Apply Cleaning ---
if 'sample_size' in df_obj1.columns:
    df_obj1['numeric_sample_size'] = df_obj1['sample_size'].apply(extract_sample_size)
else:
    df_obj1['numeric_sample_size'] = np.nan # Create column with NaNs if source is missing
    report_lines.append("Warning: 'sample_size' column missing. Cannot extract numeric sample sizes.")

# Convert 'year' to numeric, coercing errors to NaN
if 'year' in df_obj1.columns:
    df_obj1['year'] = pd.to_numeric(df_obj1['year'], errors='coerce')
    # Drop rows where year could not be parsed (optional, but good practice)
    initial_rows = len(df_obj1)
    df_obj1.dropna(subset=['year'], inplace=True)
    if len(df_obj1) < initial_rows:
        print(f"Dropped {initial_rows - len(df_obj1)} rows due to invalid 'Year' entries.")
    df_obj1['year'] = df_obj1['year'].astype(int) # Convert valid years to integer
    min_year = df_obj1['year'].min()
    max_year = df_obj1['year'].max() # Corrected from max3
    report_lines.append(f"Years covered in the valid data: {min_year} to {max_year}")
else:
    report_lines.append("Warning: 'year' column missing. Temporal analysis will not be possible.")
    min_year, max_year = None, None # Set to None if year column is missing

report_lines.append(f"Working dataset created with {len(df_obj1)} valid studies after initial cleaning.")
print(f"Data preparation complete. Working with {len(df_obj1)} studies.")

# --- 2. Effect Size Analysis ---
report_lines.append("\n2. EFFECT SIZE ANALYSIS")
report_lines.append("-" * 40)

def extract_effect_sizes(text):
    """Extracts various effect size metrics (d, g, eta^2, r) from text."""
    if pd.isna(text):
        return []

    # Define regex patterns for common effect size notations
    # Using non-capturing groups (?:...) and clear boundaries
    patterns = [
        r"Cohen's\s+d\s*=\s*(-?\d*\.?\d+)",          # Cohen's d = 0.5
        r"(?<!Cohen's\s)d\s*=\s*(-?\d*\.?\d+)",     # d = .5 (avoid matching 'sd =')
        r"Hedges'?\s+g\s*=\s*(-?\d*\.?\d+)",         # Hedges g = 0.45 or Hedges' g = 0.45
        r"(?<!Hedges'?\s)g\s*=\s*(-?\d*\.?\d+)",    # g = .45 (avoid matching 'mg =')
        r"(?:partial\s+)?eta\s+squared\s*=\s*(\d*\.?\d+)", # eta squared = 0.1, partial eta squared = .1
        r"ηp?²\s*=\s*(\d*\.?\d+)",                   # η² = 0.1, ηp² = .1 (only positive values)
        r"(?<![a-zA-Z])r\s*=\s*(-?\d*\.?\d+)",       # r = 0.3 (correlation) - ensure 'r' is not part of a word
        r"(-?\d*\.?\d+)σ",                          # e.g., -0.30σ (standard deviation units)
    ]

    text_processed = str(text).replace(',', '.')  # Standardize decimal separator
    effect_sizes = []

    for pattern in patterns:
        try:
            matches = re.findall(pattern, text_processed, re.IGNORECASE)
            for value_str in matches:
                try:
                    # Convert matched string to float
                    value_float = float(value_str)
                    # Basic plausibility check (e.g., very large d or g might be errors)
                    # Adjust range as needed based on expected values
                    if abs(value_float) < 10: # Avoid extremely large values often due to typos
                         effect_sizes.append(value_float)
                except ValueError:
                    continue # Ignore if conversion fails
        except re.error as e:
            print(f"Regex error in pattern '{pattern}': {e}")
            continue # Skip pattern if it's invalid

    # Return unique effect sizes found
    return list(set(effect_sizes))

# Extract effect sizes from potentially relevant columns (if they exist)
es_cols = ['effect_sizes', 'general_results', 'cognitive_text']
for col in es_cols:
    if col in df_obj1.columns:
        df_obj1[f'es_from_{col}'] = df_obj1[col].apply(extract_effect_sizes)
    else:
        df_obj1[f'es_from_{col}'] = [[] for _ in range(len(df_obj1))] # Empty lists if col missing

# Combine all extracted effect sizes into a single list per study
df_obj1['all_effect_sizes'] = df_obj1[[f'es_from_{col}' for col in es_cols]].sum(axis=1)
df_obj1['all_effect_sizes'] = df_obj1['all_effect_sizes'].apply(lambda x: sorted(list(set(x)))) # Unique & sorted

# Create metrics based on extracted effect sizes
df_obj1['has_effect_size'] = df_obj1['all_effect_sizes'].apply(lambda x: len(x) > 0)
df_obj1['num_effect_sizes'] = df_obj1['all_effect_sizes'].apply(len)

# Calculate mean *absolute* effect size per study (common practice for magnitude)
# Handle cases with no effect sizes (results in NaN)
df_obj1['mean_effect_size'] = df_obj1['all_effect_sizes'].apply(
    lambda x: np.mean([abs(val) for val in x]) if x else np.nan
)

# Filter to get studies that reported at least one effect size
df_with_es = df_obj1[df_obj1['has_effect_size']].copy()
num_studies_with_es = len(df_with_es)
# Flatten the list of all individual effect sizes found across studies
all_individual_es = [es for sublist in df_with_es['all_effect_sizes'] for es in sublist if pd.notna(es)]

report_lines.append(f"Studies with extractable effect sizes: {num_studies_with_es} out of {len(df_obj1)}")
if all_individual_es:
    report_lines.append(f"Total individual effect size values extracted: {len(all_individual_es)}")
else:
     report_lines.append("No individual effect size values were successfully extracted.")

mean_es, median_es = np.nan, np.nan # Initialize as NaN

if num_studies_with_es > 0 and not df_with_es['mean_effect_size'].isna().all():
    # Calculate overall mean and median effect size across studies
    mean_es = df_with_es['mean_effect_size'].mean()
    median_es = df_with_es['mean_effect_size'].median()
    min_mean_es = df_with_es['mean_effect_size'].min()
    max_mean_es = df_with_es['mean_effect_size'].max()

    report_lines.append(f"Mean absolute effect size across studies: {mean_es:.3f}")
    report_lines.append(f"Median absolute effect size across studies: {median_es:.3f}")
    report_lines.append(f"Range of mean absolute effect sizes: {min_mean_es:.3f} to {max_mean_es:.3f}")

    # Categorize mean effect sizes based on Cohen's guidelines (absolute values)
    # Bins: (-inf, 0.2), [0.2, 0.5), [0.5, 0.8), [0.8, inf)
    bins = [-np.inf, 0.2, 0.5, 0.8, np.inf]
    labels = ['Small (<0.2)', 'Medium (0.2-0.5)', 'Large (0.5-0.8)', 'Very Large (>0.8)']
    df_with_es['effect_size_category'] = pd.cut(
        df_with_es['mean_effect_size'],
        bins=bins,
        labels=labels,
        right=False # Intervals are [min, max)
    )

    # Calculate counts for each effect size category
    es_category_counts = df_with_es['effect_size_category'].value_counts().sort_index()
    report_lines.append("\nEffect size distribution by category (based on mean absolute ES per study):")
    for category, count in es_category_counts.items():
        percentage = (count / num_studies_with_es) * 100
        report_lines.append(f"  {category}: {count} studies ({percentage:.1f}%)")

    # --- Visualize Effect Size Distribution ---
    try:
        fig_es, axs = plt.subplots(2, 2, figsize=(16, 12)) # Adjusted size
        fig_es.suptitle("Effect Size Analysis", fontsize=16, y=0.99) # Adjusted title pos

        # Plot 1: Histogram of all individual effect sizes (raw values)
        if all_individual_es:
            sns.histplot(all_individual_es, kde=True, ax=axs[0, 0], bins=20)
            axs[0, 0].set_title('Distribution of All Individual Effect Sizes (Raw)')
            axs[0, 0].set_xlabel('Effect Size Value')
            axs[0, 0].set_ylabel('Frequency')
        else:
            axs[0, 0].text(0.5, 0.5, 'No individual ES data', ha='center', va='center')
            axs[0, 0].set_title('Distribution of All Individual Effect Sizes (Raw)')


        # Plot 2: Histogram of mean absolute effect sizes per study
        sns.histplot(df_with_es['mean_effect_size'].dropna(), kde=True, ax=axs[0, 1], bins=15)
        axs[0, 1].set_title('Distribution of Mean Absolute Effect Size per Study')
        axs[0, 1].set_xlabel('Mean Absolute Effect Size')
        axs[0, 1].set_ylabel('Number of Studies')
        axs[0, 1].axvline(mean_es, color='r', linestyle='--', label=f'Mean = {mean_es:.2f}')
        axs[0, 1].axvline(median_es, color='g', linestyle=':', label=f'Median = {median_es:.2f}')
        axs[0, 1].legend()

        # Plot 3: Bar chart of effect size categories
        if not es_category_counts.empty:
            sns.barplot(x=es_category_counts.index, y=es_category_counts.values, ax=axs[1, 0], palette='viridis')
            axs[1, 0].set_title('Effect Size Categories (Cohen\'s Interpretation)')
            axs[1, 0].set_xlabel('Category (Based on Mean Absolute ES)')
            axs[1, 0].set_ylabel('Number of Studies')
            axs[1, 0].tick_params(axis='x', rotation=30) # Slightly rotate labels
        else:
            axs[1, 0].text(0.5, 0.5, 'No category data', ha='center', va='center')
            axs[1, 0].set_title('Effect Size Categories')


        # Plot 4: Scatterplot of effect size vs. year (if 'year' exists and enough data)
        if 'year' in df_with_es.columns and len(df_with_es.dropna(subset=['mean_effect_size', 'year'])) > 3:
            sns.regplot(
                x='year',
                y='mean_effect_size',
                data=df_with_es,
                scatter_kws={'alpha': 0.6},
                line_kws={'color': 'red'},
                ax=axs[1, 1]
            )
            axs[1, 1].set_title('Mean Absolute Effect Size Trends by Year')
            axs[1, 1].set_xlabel('Year')
            axs[1, 1].set_ylabel('Mean Absolute Effect Size')
            # Calculate and display correlation
            corr, p_val = stats.pearsonr(df_with_es['year'].dropna(), df_with_es['mean_effect_size'].dropna())
            axs[1, 1].annotate(f'r = {corr:.2f}\np = {p_val:.2f}', xy=(0.05, 0.9), xycoords='axes fraction')
        else:
            axs[1, 1].text(0.5, 0.5, 'Insufficient data for trend analysis\n(need >3 studies with Year and ES)',
                           ha='center', va='center', fontsize=10)
            axs[1, 1].set_title('Mean Absolute Effect Size Trends by Year')

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout
        es_fig_path = os.path.join(output_dir, 'effect_size_analysis.png')
        plt.savefig(es_fig_path, dpi=100, bbox_inches='tight')
        plt.close(fig_es) # Close the figure to free memory
        report_lines.append(f"Effect size analysis visualization saved to: {es_fig_path}")
    except Exception as e:
        error_msg = f"Error creating effect size visualization: {e}"
        print(error_msg)
        report_lines.append(error_msg)

else:
    report_lines.append("Insufficient effect size data for detailed analysis or visualization.")

# --- 3. Intervention Type Analysis ---
report_lines.append("\n3. INTERVENTION TYPE ANALYSIS")
report_lines.append("-" * 40)

def categorize_intervention(text):
    """Categorizes intervention based on keywords in the description."""
    if pd.isna(text):
        return 'Unknown/Not Specified'

    text_lower = str(text).lower() # Ensure lowercase string

    # Define keywords for categories (prioritized order)
    categories = {
        'AR/VR': ['ar', 'vr', 'augment', 'virtual', 'realidad aumentada', 'realidad virtual', 'mixed reality'],
        'Game-based Learning': ['game', 'gamif', 'juego', 'serious game'],
        'Intelligent Tutor/Adaptive': ['tutor inteligente', 'intelligent tutor', 'its', 'adaptive', 'adaptativo', 'personalized learning'],
        'Simulation/Modeling': ['simulation', 'simulación', 'modelado', 'virtual lab'],
        'Robotics/Physical Computing': ['robot', 'robotics', 'physical computing', 'makeblock', 'lego mindstorms', 'scratch hardware'],
        'Software/App/Platform': ['software', 'app', 'platform', 'plataforma', 'programa', 'web-based', 'online tool'],
        'Interactive Classroom Tools': ['interactive whiteboard', 'pizarra interactiva', 'clicker', 'student response system'],
        'Programming/Coding Env': ['programming', 'coding', 'scratch', 'blockly', 'code.org'], # Added specific category
    }

    for category, keywords in categories.items():
        if any(re.search(r'\b' + re.escape(kw) + r'\b', text_lower) for kw in keywords):
            return category

    # Fallback categories
    if 'interactive' in text_lower or 'interactivo' in text_lower:
        return 'Interactive (General)'
    else:
        return 'Other/Mixed'

# Apply categorization (check if 'intervention' column exists)
if 'intervention' in df_obj1.columns:
    df_obj1['intervention_type'] = df_obj1['intervention'].apply(categorize_intervention)
else:
    df_obj1['intervention_type'] = 'Unknown/Not Specified'
    report_lines.append("Warning: 'intervention' column missing. Cannot categorize interventions.")

# Count frequencies of intervention types
intervention_counts = df_obj1['intervention_type'].value_counts()
report_lines.append("\nIntervention type distribution:")
for intervention, count in intervention_counts.items():
    percentage = (count / len(df_obj1)) * 100
    report_lines.append(f"  {intervention}: {count} studies ({percentage:.1f}%)")

# Visualize intervention type frequencies
try:
    plt.figure(figsize=(10, 6)) # Adjusted size
    sns.barplot(x=intervention_counts.values, y=intervention_counts.index, palette='Spectral')
    plt.title('Distribution of Intervention Types')
    plt.xlabel('Number of Studies')
    plt.ylabel('Intervention Type')
    plt.tight_layout()
    int_type_path = os.path.join(output_dir, 'intervention_types_distribution.png')
    plt.savefig(int_type_path, dpi=100, bbox_inches='tight')
    plt.close()
    report_lines.append(f"Intervention type distribution visualization saved to: {int_type_path}")
except Exception as e:
    error_msg = f"Error creating intervention type visualization: {e}"
    print(error_msg)
    report_lines.append(error_msg)


# Analyze effect sizes by intervention type (if effect size data is available)
es_by_intervention_filtered = pd.DataFrame() # Initialize empty dataframe

if num_studies_with_es > 0 and 'intervention_type' in df_with_es.columns:
    # Add intervention type to the dataframe with effect sizes
    df_with_es['intervention_type'] = df_obj1.loc[df_with_es.index, 'intervention_type']

    # Group by intervention type and calculate aggregate stats for mean absolute effect size
    es_by_intervention = df_with_es.groupby('intervention_type')['mean_effect_size'].agg(
        ['mean', 'median', 'std', 'count']
    ).dropna(subset=['mean']) # Drop groups where mean couldn't be calculated

    es_by_intervention = es_by_intervention.sort_values('mean', ascending=False)

    # Filter to include intervention types with a minimum number of studies for reliability
    min_studies_per_type = 2 # Adjustable threshold (e.g., 2 or 3)
    es_by_intervention_filtered = es_by_intervention[es_by_intervention['count'] >= min_studies_per_type]

    if not es_by_intervention_filtered.empty:
        report_lines.append(f"\nEffect sizes by intervention type (for types with >= {min_studies_per_type} studies):")
        for intervention, stats_row in es_by_intervention_filtered.iterrows():
            report_lines.append(f"  {intervention}: Mean Abs ES = {stats_row['mean']:.3f} (SD={stats_row['std']:.3f}), n = {int(stats_row['count'])}")

        # Visualize effect sizes by intervention type using boxplot
        try:
            plt.figure(figsize=(12, 7)) # Adjusted size
            # Filter the original df_with_es to include only the types meeting the minimum study count
            plot_data = df_with_es[df_with_es['intervention_type'].isin(es_by_intervention_filtered.index)]
            sns.boxplot(data=plot_data,
                        x='mean_effect_size', y='intervention_type',
                        order=es_by_intervention_filtered.index, # Order by mean effect size
                        palette='Spectral')
            plt.title(f'Mean Absolute Effect Size by Intervention Type (n >= {min_studies_per_type})')
            plt.xlabel('Mean Absolute Effect Size')
            plt.ylabel('Intervention Type')
            plt.axvline(mean_es, color='grey', linestyle='--', label=f'Overall Mean ES ({mean_es:.2f})')
            plt.legend(loc='lower right')
            plt.tight_layout()

            es_by_int_path = os.path.join(output_dir, 'effect_size_by_intervention.png')
            plt.savefig(es_by_int_path, dpi=100, bbox_inches='tight')
            plt.close()
            report_lines.append(f"Effect size by intervention visualization saved to: {es_by_int_path}")
        except Exception as e:
            error_msg = f"Error creating ES by intervention visualization: {e}"
            print(error_msg)
            report_lines.append(error_msg)

        # Perform statistical test (e.g., Kruskal-Wallis) if enough groups meet the criteria
        if len(es_by_intervention_filtered) >= 2: # Need at least 2 groups to compare
            # Prepare data for the test: list of effect sizes for each intervention type
            groups_for_test = [
                df_with_es[df_with_es['intervention_type'] == int_type]['mean_effect_size'].dropna().values
                for int_type in es_by_intervention_filtered.index
            ]
            # Ensure all groups have data
            groups_for_test = [g for g in groups_for_test if len(g) > 0]

            if len(groups_for_test) >= 2:
                try:
                    stat, p_value = stats.kruskal(*groups_for_test) # Non-parametric test suitable for small/non-normal samples
                    report_lines.append(f"\nKruskal-Wallis test for differences in mean absolute ES between intervention types:")
                    report_lines.append(f"  H-statistic: {stat:.3f}, p-value: {p_value:.4f}")
                    if p_value < 0.05:
                        report_lines.append("  Result: Significant differences detected (p < 0.05). Post-hoc tests would be needed to identify specific pairs.")
                    else:
                        report_lines.append("  Result: No significant differences detected (p >= 0.05).")
                except Exception as e:
                    report_lines.append(f"  Error performing Kruskal-Wallis test: {str(e)}")
            else:
                 report_lines.append("\nInsufficient valid groups for Kruskal-Wallis test.")
    else:
        report_lines.append(f"\nInsufficient data for comparing effect sizes across intervention types (minimum {min_studies_per_type} studies required per type).")
else:
    report_lines.append("\nEffect size analysis by intervention type skipped due to lack of sufficient effect size data.")


# --- 4. Mathematical Domain Analysis ---
report_lines.append("\n4. MATHEMATICAL DOMAIN ANALYSIS")
report_lines.append("-" * 40)

# Define mathematical domains and associated keywords (Spanish/English)
# Expanded and refined keywords
math_domains = {
    'Numeracy/Number Sense': ['numeracy', 'number sense', 'counting', 'numerical', 'número', 'conteo', 'numérico', 'sentido numérico', 'magnitude comparison', 'subitizing'],
    'Arithmetic/Calculation': ['arithmetic', 'calculation', 'addition', 'subtraction', 'multiplication', 'division', 'cálculo', 'suma', 'resta', 'multiplicación', 'división', 'operaciones', 'fluency', 'math facts', 'mental math'],
    'Geometry/Spatial': ['geometry', 'spatial', 'shape', 'geometric', 'geometría', 'espacial', 'forma', 'visualization', 'mapping', 'rotation'],
    'Algebra/Pre-Algebra': ['algebra', 'equation', 'variable', 'ecuación', 'expresión', 'algebraic thinking', 'patterns', 'functions', 'pre-algebra'],
    'Problem Solving': ['problem solving', 'problem-solving', 'resolución de problemas', 'word problems', 'problemas verbales', 'mathematical reasoning', 'heuristic'],
    'Computational Thinking': ['computational thinking', 'algorithm', 'programming', 'coding', 'pensamiento computacional', 'algoritmo', 'programación', 'logical thinking', 'debugging', 'decomposition', 'pattern recognition', 'abstraction'],
    'Fractions/Decimals/Ratio': ['fraction', 'decimal', 'ratio', 'proportion', 'percent', 'fracción', 'proporción', 'razón', 'porcentaje', 'rational numbers'],
    'Measurement': ['measurement', 'measure', 'medición', 'medida', 'units', 'unidades', 'length', 'area', 'volume', 'time', 'money'],
    'Data Analysis/Stats/Prob': ['statistic', 'probability', 'data analysis', 'estadística', 'probabilidad', 'datos', 'data handling', 'graph', 'chart', 'gráfica', 'average', 'chance']
}

def identify_domains(text):
    """Identifies mathematical domains mentioned in text using keywords."""
    if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
        return []

    text_lower = text.lower()
    found_domains = set() # Use a set to automatically handle duplicates

    for domain, keywords in math_domains.items():
        for keyword in keywords:
            # Use word boundaries (\b) to match whole words only
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                found_domains.add(domain)
                break # Move to the next domain once one keyword is found for the current domain

    return sorted(list(found_domains)) # Return sorted list

# Apply domain identification to the combined cognitive text (if column exists)
if 'cognitive_text' in df_obj1.columns:
    df_obj1['identified_domains'] = df_obj1['cognitive_text'].apply(identify_domains)
    df_obj1['num_domains'] = df_obj1['identified_domains'].apply(len)
else:
    df_obj1['identified_domains'] = [[] for _ in range(len(df_obj1))]
    df_obj1['num_domains'] = 0
    report_lines.append("Warning: 'cognitive_text' column missing. Cannot identify mathematical domains.")

# Calculate domain frequencies across all studies
all_identified_domains = [domain for sublist in df_obj1['identified_domains'] for domain in sublist]
domain_counts = pd.Series(all_identified_domains).value_counts()

domain_df = pd.DataFrame({'Domain': domain_counts.index, 'Frequency': domain_counts.values})
# domain_df = domain_df.sort_values('Frequency', ascending=False) # Already sorted by value_counts

if not domain_df.empty:
    report_lines.append("\nMathematical domain frequency across studies:")
    for _, row in domain_df.iterrows():
        percentage = (row['Frequency'] / len(df_obj1)) * 100 # Percentage of studies mentioning the domain
        report_lines.append(f"  {row['Domain']}: {row['Frequency']} studies ({percentage:.1f}%)")

    # Visualize domain frequencies
    try:
        plt.figure(figsize=(12, max(6, len(domain_df) * 0.5))) # Adjust height based on number of domains
        sns.barplot(data=domain_df, x='Frequency', y='Domain', palette='Blues_r', orient='h')
        plt.title('Frequency of Mathematical Domains Addressed in Studies')
        plt.xlabel('Number of Studies')
        plt.ylabel('Mathematical Domain')
        plt.tight_layout()

        domains_path = os.path.join(output_dir, 'mathematical_domains_frequency.png')
        plt.savefig(domains_path, dpi=100, bbox_inches='tight')
        plt.close()
        report_lines.append(f"Mathematical domains visualization saved to: {domains_path}")
    except Exception as e:
        error_msg = f"Error creating mathematical domains visualization: {e}"
        print(error_msg)
        report_lines.append(error_msg)

else:
    report_lines.append("No mathematical domains were identified in the provided text data.")


# --- 5. Text Analysis of Cognitive Outcomes ---
report_lines.append("\n5. TEXT ANALYSIS OF COGNITIVE OUTCOMES")
report_lines.append("-" * 40)

def preprocess_text(text):
    """Preprocesses text for NLP: lowercase, remove punctuation/numbers, tokenize, remove stopwords."""
    if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
        return ''

    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation and numbers, keep letters and spaces (incl. Spanish characters)
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)
    # 3. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 4. Tokenize
    tokens = word_tokenize(text)

    # 5. Remove stopwords (English and Spanish)
    try:
        stop_words_en = set(stopwords.words('english'))
        stop_words_es = set(stopwords.words('spanish'))
        stop_words = stop_words_en.union(stop_words_es)
    except Exception as e:
        print(f"Warning: Could not load stopwords - {e}. Proceeding without stopword removal.")
        stop_words = set()


    # Add custom stopwords relevant to reviews/education context
    custom_stopwords = {
        'use', 'using', 'used', 'study', 'studies', 'group', 'groups', 'result', 'results',
        'effect', 'effects', 'significant', 'significantly', 'found', 'showed', 'increased', 'improved',
        'uso', 'usar', 'utilizar', 'estudio', 'estudios', 'grupo', 'grupos', 'resultado',
        'resultados', 'efecto', 'efectos', 'significativo', 'significativa', 'encontrado',
        'mostró', 'incrementado', 'mejorado', 'investigación', 'research', 'analysis', 'análisis',
        'intervention', 'control', 'comparison', 'students', 'teachers', 'participants',
        'intervención', 'comparación', 'estudiantes', 'profesores', 'participantes',
        'digital', 'tool', 'tools', 'technology', 'technologies', 'herramienta', 'herramientas',
        'tecnología', 'tecnologías', 'skill', 'skills', 'habilidad', 'habilidades', 'level', 'levels',
        'primary', 'education', 'school', 'primaria', 'educación', 'escuela', 'impact', 'influence',
        'impacto', 'influencia', 'development', 'desarrollo', 'learning', 'aprendizaje',
        # Add short common words often missed
        'also', 'however', 'may', 'might', 'could', 'would', 'one', 'two', 'post', 'pre'
        }
    stop_words = stop_words.union(custom_stopwords)

    # 6. Filter tokens: remove stopwords and short tokens
    processed_tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

    return ' '.join(processed_tokens)

# Apply text preprocessing to the combined cognitive text (if column exists)
valid_texts = []
if 'cognitive_text' in df_obj1.columns:
    df_obj1['processed_cognitive_text'] = df_obj1['cognitive_text'].apply(preprocess_text)
    # Get a list of non-empty processed texts for analysis
    valid_texts = df_obj1['processed_cognitive_text'].dropna().tolist()
    valid_texts = [text for text in valid_texts if text.strip()] # Ensure no empty strings remain
    report_lines.append(f"Preprocessed cognitive text for {len(valid_texts)} studies.")
else:
    df_obj1['processed_cognitive_text'] = ''
    report_lines.append("Warning: 'cognitive_text' column missing. Skipping text analysis.")

term_importance = pd.DataFrame() # Initialize

# Proceed with TF-IDF and Word Cloud only if there's enough processed text data
if len(valid_texts) > 3: # Need a few documents for TF-IDF to be meaningful
    try:
        # --- TF-IDF Analysis ---
        # Use uni-grams and bi-grams, limit features to top 50
        tfidf_vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2), stop_words=None) # Stopwords already removed
        tfidf_matrix = tfidf_vectorizer.fit_transform(valid_texts)

        # Get feature names (terms) and their summed TF-IDF scores (importance)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        total_importance = tfidf_matrix.sum(axis=0).A1 # .A1 converts matrix row to flat numpy array

        # Create a DataFrame for easy sorting and plotting
        term_importance = pd.DataFrame({'term': feature_names, 'importance': total_importance})
        term_importance = term_importance.sort_values('importance', ascending=False).reset_index(drop=True)

        report_lines.append("\nTop 15 key terms/phrases in cognitive effectiveness descriptions (TF-IDF):")
        for _, row in term_importance.head(15).iterrows():
            report_lines.append(f"  {row['term']} (Score: {row['importance']:.3f})")

        # --- Visualization: Key Terms Bar Chart and Word Cloud ---
        fig_text, axs = plt.subplots(2, 1, figsize=(14, 12)) # Adjusted layout
        fig_text.suptitle("Text Analysis of Cognitive Outcome Descriptions", fontsize=16)

        # Bar chart of top TF-IDF terms
        if not term_importance.empty:
            sns.barplot(data=term_importance.head(15), x='importance', y='term', palette='viridis', ax=axs[0])
            axs[0].set_title('Top 15 Key Terms/Phrases (TF-IDF Importance)')
            axs[0].set_xlabel('Total TF-IDF Score')
            axs[0].set_ylabel('Term / Phrase')
        else:
             axs[0].text(0.5, 0.5, 'No terms found', ha='center', va='center')
             axs[0].set_title('Top Key Terms/Phrases (TF-IDF Importance)')

        # Word cloud
        all_processed_text = ' '.join(valid_texts)
        if all_processed_text.strip(): # Check if there is any text left after processing
            try:
                wordcloud = WordCloud(width=1000, height=500, # Larger size
                                      background_color='white',
                                      colormap='viridis', # Consistent palette
                                      max_words=75, # Show more words
                                      contour_width=1,
                                      contour_color='steelblue',
                                      collocations=False, # Avoid generating bi-grams within wordcloud itself
                                      random_state=42 # for reproducibility
                                      ).generate(all_processed_text)

                axs[1].imshow(wordcloud, interpolation='bilinear')
                axs[1].axis('off')
                axs[1].set_title('Word Cloud of Preprocessed Key Terms')
            except Exception as wc_error:
                 axs[1].text(0.5, 0.5, f'Word cloud generation failed:\n{wc_error}', ha='center', va='center')
                 axs[1].set_title('Word Cloud (Error)')
                 axs[1].axis('off')
                 print(f"Word cloud generation failed: {wc_error}")
        else:
            axs[1].text(0.5, 0.5, 'Insufficient unique text data\n after preprocessing for word cloud',
                       ha='center', va='center', fontsize=12)
            axs[1].set_title('Word Cloud (Insufficient Data)')
            axs[1].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        text_analysis_path = os.path.join(output_dir, 'cognitive_text_analysis.png')
        plt.savefig(text_analysis_path, dpi=100, bbox_inches='tight')
        plt.close(fig_text)
        report_lines.append(f"Text analysis visualization saved to: {text_analysis_path}")

    except Exception as e:
        error_msg = f"Error during text analysis (TF-IDF/WordCloud): {str(e)}"
        print(error_msg)
        report_lines.append(error_msg)
else:
    report_lines.append("Insufficient valid text data (need >3 studies with cognitive descriptions) for meaningful NLP analysis (TF-IDF, Word Cloud).")

# --- 6. Relationship Network Analysis (Intervention Type <-> Math Domain) ---
report_lines.append("\n6. RELATIONSHIP NETWORK ANALYSIS")
report_lines.append("-" * 40)

co_occurrences = []
pair_frequencies = pd.Series(dtype=int) # Initialize empty Series
G = nx.Graph() # Initialize empty graph

# Check if necessary columns exist
if 'intervention_type' in df_obj1.columns and 'identified_domains' in df_obj1.columns:
    # Create pairs of (intervention_type, domain) for each study
    for _, row in df_obj1.iterrows():
        intervention = row['intervention_type']
        domains = row['identified_domains']

        # Include pairs only if intervention is known and domains were identified
        if intervention != 'Unknown/Not Specified' and domains:
            for domain in domains:
                # Ensure domain is a non-empty string
                if isinstance(domain, str) and domain.strip():
                     co_occurrences.append((intervention, domain))

    # Count the frequency of each unique (intervention, domain) pair
    if co_occurrences:
        pair_frequencies = pd.Series(co_occurrences).value_counts()

        report_lines.append("\nMost frequent intervention-domain relationships (co-occurrence count):")
        for (intervention, domain), freq in pair_frequencies.head(10).items():
            report_lines.append(f"  {intervention} → {domain}: {freq} studies")

        # --- Create and Visualize the Network Graph ---
        # Add nodes: interventions and domains
        interventions_in_network = set(pair[0] for pair in pair_frequencies.index)
        domains_in_network = set(pair[1] for pair in pair_frequencies.index)

        for intervention in interventions_in_network:
            G.add_node(intervention, type='Intervention', bipartite=0) # Add attributes for coloring/layout

        for domain in domains_in_network:
            G.add_node(domain, type='Domain', bipartite=1)

        # Add edges with weights based on co-occurrence frequency
        for (intervention, domain), weight in pair_frequencies.items():
            # Add edge only if both nodes exist (safety check)
            if G.has_node(intervention) and G.has_node(domain):
                G.add_edge(intervention, domain, weight=weight)

        if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
            try:
                plt.figure(figsize=(18, 14)) # Larger figure for clarity

                # Use a layout algorithm suitable for bipartite graphs if desired, or spring layout
                # pos = nx.bipartite_layout(G, interventions_in_network)
                pos = nx.spring_layout(G, k=0.6, iterations=70, seed=42) # Spring layout often works well visually

                # Node styling
                node_colors = ['skyblue' if G.nodes[n]['type'] == 'Intervention' else 'lightcoral' for n in G.nodes()]
                # Size nodes based on degree (number of connections), with a base size
                node_degrees = dict(G.degree())
                node_sizes = [(node_degrees.get(n, 0) * 100) + 500 for n in G.nodes()] # Adjust multiplier as needed

                # Edge styling
                # Normalize edge weights for better visualization if weights vary widely
                max_weight = max(nx.get_edge_attributes(G, 'weight').values()) if G.edges else 1
                edge_widths = [G.edges[u, v]['weight'] / max_weight * 5 + 0.5 for u, v in G.edges()] # Adjust multiplier/base

                # Draw the network components
                nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
                nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='grey')
                nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

                plt.title('Network of Co-occurrence: Intervention Types and Mathematical Domains', size=18)

                # Create legend handles manually
                intervention_patch = plt.Line2D([], [], marker='o', markersize=10, linewidth=0, color='skyblue', label='Intervention Type')
                domain_patch = plt.Line2D([], [], marker='o', markersize=10, linewidth=0, color='lightcoral', label='Mathematical Domain')
                plt.legend(handles=[intervention_patch, domain_patch], loc='upper right', fontsize=12)

                plt.axis('off') # Hide axes
                plt.tight_layout()

                network_path = os.path.join(output_dir, 'intervention_domain_network.png')
                plt.savefig(network_path, dpi=100, bbox_inches='tight')
                plt.close()
                report_lines.append(f"Relationship network visualization saved to: {network_path}")

                # --- Network Metrics ---
                report_lines.append("\nNetwork Analysis Metrics:")

                # Degree Centrality (most connected nodes)
                degree_centrality = nx.degree_centrality(G)
                # Sort by centrality score, descending
                sorted_centrality = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)

                report_lines.append("  Most connected nodes (Normalized Degree Centrality):")
                for node, centrality in sorted_centrality[:5]: # Top 5
                     node_type = G.nodes[node]['type']
                     report_lines.append(f"    - {node} ({node_type}): {centrality:.3f}")

                # Betweenness Centrality (nodes acting as bridges) - calculated only if graph is large enough
                if G.number_of_nodes() > 2:
                    try:
                        betweenness = nx.betweenness_centrality(G, normalized=True)
                        sorted_betweenness = sorted(betweenness.items(), key=lambda item: item[1], reverse=True)

                        report_lines.append("  Key bridge nodes (Normalized Betweenness Centrality):")
                        # Report top nodes with non-zero betweenness
                        reported_count = 0
                        for node, score in sorted_betweenness:
                            if score > 1e-6 and reported_count < 5: # Check for non-negligible score, limit to top 5
                                node_type = G.nodes[node]['type']
                                report_lines.append(f"    - {node} ({node_type}): {score:.3f}")
                                reported_count += 1
                        if reported_count == 0:
                            report_lines.append("    - No significant bridge nodes found.")
                    except Exception as e:
                        report_lines.append(f"    - Error calculating betweenness centrality: {e}")


            except Exception as e:
                error_msg = f"Error creating network visualization or calculating metrics: {e}"
                print(error_msg)
                report_lines.append(error_msg)
        else:
             report_lines.append("Network graph could not be generated (no nodes or edges).")
    else:
        report_lines.append("No co-occurrences found between known intervention types and identified domains. Skipping network analysis.")
else:
    report_lines.append("Skipping Relationship Network Analysis because 'intervention_type' or 'identified_domains' columns are missing.")


# --- 7. Time Trend Analysis ---
report_lines.append("\n7. TEMPORAL TREND ANALYSIS")
report_lines.append("-" * 40)

yearly_metrics = pd.DataFrame() # Initialize

# Check if 'year' column exists and is numeric
if 'year' in df_obj1.columns and pd.api.types.is_numeric_dtype(df_obj1['year']):
    # Define aggregations
    agg_dict = {
        'study_count': ('year', 'size'), # Count studies per year
    }
    # Add optional aggregations if columns exist
    if 'numeric_sample_size' in df_obj1.columns:
        agg_dict['mean_sample_size'] = ('numeric_sample_size', 'mean')
    if 'num_domains' in df_obj1.columns:
         agg_dict['mean_domains_per_study'] = ('num_domains', 'mean')

    yearly_metrics = df_obj1.groupby('year').agg(**agg_dict)

    # Add mean effect size per year if available
    if num_studies_with_es > 0 and 'year' in df_with_es.columns:
        yearly_es = df_with_es.groupby('year')['mean_effect_size'].agg(['mean', 'count'])
        yearly_es.rename(columns={'mean': 'mean_effect_size', 'count': 'es_study_count'}, inplace=True)
        yearly_metrics = yearly_metrics.join(yearly_es, how='left') # Left join to keep all years

    # Ensure index is sorted
    yearly_metrics.sort_index(inplace=True)

    report_lines.append("\nTemporal distribution of studies:")
    for year, row in yearly_metrics.iterrows():
        report_lines.append(f"  {int(year)}: {int(row['study_count'])} studies")
        if 'mean_sample_size' in row and pd.notna(row['mean_sample_size']):
             report_lines[-1] += f", Mean Sample Size: {row['mean_sample_size']:.1f}"
        if 'mean_effect_size' in row and pd.notna(row['mean_effect_size']):
             report_lines[-1] += f", Mean Abs ES: {row['mean_effect_size']:.3f} (n={int(row.get('es_study_count',0))})"


    # Visualize trends if there are enough years of data
    if len(yearly_metrics) > 2:
        try:
            # Determine number of plots needed
            plot_count = 1 # Always plot study count
            if 'mean_effect_size' in yearly_metrics.columns and yearly_metrics['mean_effect_size'].notna().sum() > 1:
                plot_count = 2

            fig_trends, axs = plt.subplots(plot_count, 1, figsize=(14, 6 * plot_count), sharex=True) # Share X axis
            if plot_count == 1: # If only one plot, axs is not an array
                axs = [axs]
            fig_trends.suptitle("Temporal Trends in Interactive Math Tools Research", fontsize=16, y=0.99)

            # Plot 1: Studies per year (Bar chart)
            axs[0].bar(yearly_metrics.index, yearly_metrics['study_count'], color='steelblue')
            axs[0].set_title('Number of Studies Published per Year')
            # axs[0].set_xlabel('Year') # X label only on bottom plot
            axs[0].set_ylabel('Number of Studies')
            axs[0].grid(axis='y', linestyle='--', alpha=0.7)
            # Add a simple trend line for study count
            if len(yearly_metrics) > 2:
                 x_years = np.array(yearly_metrics.index)
                 y_counts = np.array(yearly_metrics['study_count'])
                 z_counts = np.polyfit(x_years, y_counts, 1)
                 p_counts = np.poly1d(z_counts)
                 axs[0].plot(x_years, p_counts(x_years), "r--", alpha=0.6, label=f"Trend (Slope={z_counts[0]:.2f})")
                 axs[0].legend()


            # Plot 2: Mean Effect Size trend over time (if available)
            z_es = None # Initialize slope variable
            if plot_count == 2:
                years_with_es = yearly_metrics[yearly_metrics['mean_effect_size'].notna()]
                if len(years_with_es) > 1:
                    axs[1].plot(years_with_es.index, years_with_es['mean_effect_size'],
                              marker='o', linestyle='-', color='darkred', linewidth=2, label='Mean Abs ES')
                    axs[1].set_title('Trend of Mean Absolute Effect Size per Year')
                    axs[1].set_xlabel('Year')
                    axs[1].set_ylabel('Mean Absolute Effect Size')
                    axs[1].grid(axis='y', linestyle='--', alpha=0.7)

                    # Add linear trendline if enough data points (>2)
                    if len(years_with_es) > 2:
                        x_es_years = np.array(years_with_es.index)
                        y_es_means = np.array(years_with_es['mean_effect_size'])
                        z_es = np.polyfit(x_es_years, y_es_means, 1) # Calculate slope and intercept
                        p_es = np.poly1d(z_es)
                        axs[1].plot(x_es_years, p_es(x_es_years), "r--", alpha=0.7, label=f"Trend (Slope={z_es[0]:.3f})")
                        axs[1].legend()

                        # Report trend information based on slope
                        report_lines.append(f"\nMean Absolute Effect Size trend over time (linear fit): Slope = {z_es[0]:.3f}")
                        if abs(z_es[0]) < 0.005: # Threshold for "stable"
                            trend_direction = "relatively stable"
                        elif z_es[0] > 0:
                            trend_direction = "increasing"
                        else:
                            trend_direction = "decreasing"
                        report_lines.append(f"  Overall direction: {trend_direction}")
                    else:
                        axs[1].text(0.5, 0.5, 'Need >2 years with ES data for trendline',
                                  ha='center', va='center', transform=axs[1].transAxes)
                        axs[1].legend()
                else:
                     axs[1].text(0.5, 0.5, 'Insufficient effect size data points (<2) for trend plot',
                               ha='center', va='center', transform=axs[1].transAxes)
                     axs[1].set_title('Trend of Mean Absolute Effect Size per Year')
                     axs[1].set_xlabel('Year')


            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to prevent title overlap
            trends_path = os.path.join(output_dir, 'temporal_trends.png')
            plt.savefig(trends_path, dpi=100, bbox_inches='tight')
            plt.close(fig_trends)
            report_lines.append(f"Temporal trend visualization saved to: {trends_path}")
        except Exception as e:
            error_msg = f"Error creating temporal trends visualization: {e}"
            print(error_msg)
            report_lines.append(error_msg)
    else:
        report_lines.append("Insufficient temporal data (<= 2 years) for meaningful trend visualization.")
else:
    report_lines.append("Temporal Trend Analysis skipped because 'year' column is missing or not numeric.")


# --- 8. Summary Dashboard ---
report_lines.append("\n8. SUMMARY DASHBOARD")
report_lines.append("-" * 40)
print("Generating Summary Dashboard...")

try:
    # Create a dashboard figure with a grid layout
    fig_dash = plt.figure(figsize=(22, 18)) # Larger figure for dashboard
    # GridSpec: 3 rows, 3 columns. Network plot spans bottom row.
    gs = fig_dash.add_gridspec(3, 3, height_ratios=[1, 1, 1.5], hspace=0.45, wspace=0.35)
    fig_dash.suptitle('Interactive Digital Tools in Math Education: Key Findings Summary', fontsize=24, y=0.98)

    # --- Panel 1: Top Intervention Types (Top Left) ---
    ax1 = fig_dash.add_subplot(gs[0, 0])
    if not intervention_counts.empty:
        intervention_counts_sorted = intervention_counts.nlargest(6) # Show top 6
        sns.barplot(x=intervention_counts_sorted.values, y=intervention_counts_sorted.index, palette='Blues', ax=ax1, orient='h')
        ax1.set_title('Top Intervention Types Used', fontsize=14)
        ax1.set_xlabel('Number of Studies', fontsize=12)
        ax1.tick_params(axis='y', labelsize=10)
    else:
        ax1.text(0.5, 0.5, 'No intervention data', ha='center', va='center')
        ax1.set_title('Top Intervention Types Used', fontsize=14)
    ax1.grid(axis='x', linestyle='--', alpha=0.6)


    # --- Panel 2: Studies by Year (Top Middle) ---
    ax2 = fig_dash.add_subplot(gs[0, 1])
    if not yearly_metrics.empty and 'study_count' in yearly_metrics.columns:
        year_counts = yearly_metrics['study_count']
        ax2.bar(year_counts.index, year_counts.values, color='steelblue')
        ax2.set_title('Studies Published by Year', fontsize=14)
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Number of Studies', fontsize=12)
        # Optional: add trend line if desired
        if len(year_counts) > 2:
             x_yr = np.array(year_counts.index)
             y_ct = np.array(year_counts.values)
             z_ct = np.polyfit(x_yr, y_ct, 1)
             p_ct = np.poly1d(z_ct)
             ax2.plot(x_yr, p_ct(x_yr), "r--", alpha=0.7)
    else:
        ax2.text(0.5, 0.5, 'No year data', ha='center', va='center')
        ax2.set_title('Studies Published by Year', fontsize=14)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    # --- Panel 3: Top Mathematical Domains (Top Right) ---
    ax3 = fig_dash.add_subplot(gs[0, 2])
    if not domain_df.empty:
        top_domains = domain_df.head(6) # Show top 6
        sns.barplot(x='Frequency', y='Domain', data=top_domains, palette='Greens', ax=ax3, orient='h')
        ax3.set_title('Top Mathematical Domains Addressed', fontsize=14)
        ax3.set_xlabel('Number of Studies', fontsize=12)
        ax3.tick_params(axis='y', labelsize=10)
    else:
        ax3.text(0.5, 0.5, 'No domain data available', ha='center', va='center')
        ax3.set_title('Top Mathematical Domains Addressed', fontsize=14)
    ax3.grid(axis='x', linestyle='--', alpha=0.6)


    # --- Panel 4: Effect Size Distribution (Middle Left) ---
    ax4 = fig_dash.add_subplot(gs[1, 0])
    if num_studies_with_es > 0 and 'effect_size_category' in df_with_es.columns:
        # Use the previously calculated category counts
        if 'es_category_counts' in locals() and not es_category_counts.empty:
             sns.barplot(x=es_category_counts.index, y=es_category_counts.values, palette='Oranges', ax=ax4)
             ax4.set_title('Distribution of Effect Size Magnitudes', fontsize=14)
             ax4.set_xlabel('Effect Size Category (Mean Abs ES)', fontsize=12)
             ax4.set_ylabel('Number of Studies', fontsize=12)
             ax4.tick_params(axis='x', rotation=25, labelsize=10) # Rotate labels slightly
        else:
             ax4.text(0.5, 0.5, 'Categorization failed', ha='center', va='center')
             ax4.set_title('Distribution of Effect Size Magnitudes', fontsize=14)
    else:
        ax4.text(0.5, 0.5, 'No effect size data available', ha='center', va='center')
        ax4.set_title('Distribution of Effect Size Magnitudes', fontsize=14)
    ax4.grid(axis='y', linestyle='--', alpha=0.6)


    # --- Panel 5: Effect Size by Intervention Type (Middle Middle) ---
    ax5 = fig_dash.add_subplot(gs[1, 1])
    # Use the filtered data calculated earlier (es_by_intervention_filtered)
    if not es_by_intervention_filtered.empty:
        # Plot mean effect size for types with enough studies
        data_to_plot = es_by_intervention_filtered.reset_index()
        sns.barplot(
            x='mean', # Mean absolute effect size
            y='intervention_type',
            data=data_to_plot,
            palette='Reds',
            ax=ax5,
            orient='h'
        )
        # Add error bars if std is available
        if 'std' in data_to_plot.columns:
             ax5.errorbar(x=data_to_plot['mean'], y=data_to_plot['intervention_type'],
                          xerr=data_to_plot['std'], fmt='none', c='black', capsize=3, alpha=0.6)

        ax5.set_title(f'Mean Abs ES by Intervention (n>={min_studies_per_type})', fontsize=14)
        ax5.set_xlabel('Mean Absolute Effect Size', fontsize=12)
        ax5.set_ylabel('') # Remove y-label for cleaner look
        ax5.tick_params(axis='y', labelsize=10)
        ax5.axvline(mean_es, color='grey', linestyle='--', label=f'Overall Mean ({mean_es:.2f})')
        ax5.legend(fontsize=10)
    else:
        ax5.text(0.5, 0.5, 'Insufficient data for\nES by intervention comparison', ha='center', va='center')
        ax5.set_title('Mean Abs ES by Intervention', fontsize=14)
    ax5.grid(axis='x', linestyle='--', alpha=0.6)


    # --- Panel 6: Word Cloud of Key Terms (Middle Right) ---
    ax6 = fig_dash.add_subplot(gs[1, 2])
    # Use the text processed earlier
    all_processed_text_dash = ' '.join(valid_texts)
    if all_processed_text_dash.strip():
        try:
            wordcloud_dash = WordCloud(width=400, height=300,
                                   background_color='white',
                                   colormap='viridis',
                                   max_words=40, # Fewer words for small panel
                                   contour_width=1,
                                   collocations=False,
                                   random_state=42
                                   ).generate(all_processed_text_dash)
            ax6.imshow(wordcloud_dash, interpolation='bilinear')
            ax6.set_title('Key Terms in Cognitive Outcomes', fontsize=14)
        except Exception as e:
             ax6.text(0.5, 0.5, f'WordCloud Error:\n{e}', ha='center', va='center', fontsize=9)
             ax6.set_title('Key Terms (Error)', fontsize=14)
    else:
        ax6.text(0.5, 0.5, 'Insufficient text data', ha='center', va='center')
        ax6.set_title('Key Terms in Cognitive Outcomes', fontsize=14)
    ax6.axis('off')


    # --- Panel 7: Simplified Network Visualization (Bottom Row, Spanning All Columns) ---
    ax7 = fig_dash.add_subplot(gs[2, :])
    # Use the graph G created earlier
    if G.number_of_nodes() > 1 and G.number_of_edges() > 0:
        try:
            # Use a layout suitable for the potentially smaller space
            pos_dash = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

            # Node styling (similar to previous network plot, maybe smaller sizes)
            node_colors_dash = ['skyblue' if G.nodes[n]['type'] == 'Intervention' else 'lightcoral' for n in G.nodes()]
            node_degrees_dash = dict(G.degree())
            node_sizes_dash = [(node_degrees_dash.get(n, 0) * 60) + 250 for n in G.nodes()] # Smaller sizes

            # Edge styling (similar to previous, maybe thinner lines)
            max_weight_dash = max(nx.get_edge_attributes(G, 'weight').values()) if G.edges else 1
            edge_widths_dash = [G.edges[u, v]['weight'] / max_weight_dash * 3 + 0.3 for u, v in G.edges()]

            # Draw network
            nx.draw_networkx_nodes(G, pos_dash, node_size=node_sizes_dash, node_color=node_colors_dash, alpha=0.85, ax=ax7)
            nx.draw_networkx_edges(G, pos_dash, width=edge_widths_dash, alpha=0.5, edge_color='gray', ax=ax7)
            nx.draw_networkx_labels(G, pos_dash, font_size=9, font_weight='normal', ax=ax7) # Slightly smaller font

            ax7.set_title('Relationship Network: Intervention Types and Mathematical Domains', size=16)
            # Add legend (re-using handles from earlier)
            if 'intervention_patch' in locals() and 'domain_patch' in locals():
                 ax7.legend(handles=[intervention_patch, domain_patch], loc='upper right', fontsize=11)
            ax7.axis('off')

        except Exception as e:
             ax7.text(0.5, 0.5, f'Network plot error: {e}', ha='center', va='center', fontsize=12)
             ax7.set_title('Relationship Network (Error)', size=16)
             ax7.axis('off')
    else:
        ax7.text(0.5, 0.5, 'Insufficient data for network visualization', ha='center', va='center', fontsize=14)
        ax7.set_title('Relationship Network', size=16)
        ax7.axis('off')

    # --- Final Adjustments and Saving ---
    plt.tight_layout(rect=[0, 0.02, 1, 0.96]) # Adjust layout rect to prevent overlap
    dashboard_path = os.path.join(output_dir, 'summary_dashboard.png')
    plt.savefig(dashboard_path, dpi=120, bbox_inches='tight') # Higher DPI for dashboard
    plt.close(fig_dash)
    report_lines.append(f"Summary dashboard visualization saved to: {dashboard_path}")
    print("Summary Dashboard generated successfully.")

except Exception as e:
    error_msg = f"FATAL ERROR generating the summary dashboard: {e}"
    print(error_msg)
    report_lines.append(f"\n!! ERROR: Summary dashboard could not be generated due to: {e} !!")


# --- 9. Key Findings and Conclusions Synthesis ---
report_lines.append("\n9. KEY FINDINGS AND CONCLUSIONS")
report_lines.append("-" * 40)

# Summarize key findings based on the analyses performed
total_studies_analyzed = len(df_obj1)
report_lines.append(f"\nThis analysis examined {total_studies_analyzed} studies focusing on the impact of interactive digital tools")
report_lines.append(f"on mathematical skills and computational thinking in primary education.")
if min_year and max_year:
    report_lines.append(f"The studies cover the period from {min_year} to {max_year}.")

# --- Effectiveness Findings (based on Effect Sizes) ---
report_lines.append("\n1. Effectiveness of Interactive Digital Tools:")
if num_studies_with_es > 0 and not np.isnan(mean_es):
    report_lines.append(f"  - {num_studies_with_es} out of {total_studies_analyzed} studies ({num_studies_with_es/total_studies_analyzed:.1%}) provided extractable effect sizes.")
    report_lines.append(f"  - The average absolute effect size across these studies was {mean_es:.3f} (Median: {median_es:.3f}).")

    # Interpret overall effect size magnitude
    if mean_es < 0.2: effect_interpretation = "small"
    elif mean_es < 0.5: effect_interpretation = "medium"
    elif mean_es < 0.8: effect_interpretation = "large"
    else: effect_interpretation = "very large"
    report_lines.append(f"  - This suggests an overall {effect_interpretation} positive impact on average.")

    # Summary based on categories
    if 'es_category_counts' in locals() and not es_category_counts.empty:
        main_category = es_category_counts.idxmax()
        main_percentage = (es_category_counts.max() / es_category_counts.sum()) * 100
        report_lines.append(f"  - The most common effect size magnitude found was '{main_category}', observed in {main_percentage:.1f}% of studies with ES data.")

    # Most effective intervention types based on ES
    if not es_by_intervention_filtered.empty:
         # Report top 1-2 based on mean ES
         top_interventions = es_by_intervention_filtered.head(2)
         report_lines.append(f"  - Intervention types showing the highest average effectiveness (min. {min_studies_per_type} studies):")
         for int_name, stats in top_interventions.iterrows():
              report_lines.append(f"      * {int_name} (Mean Abs ES = {stats['mean']:.3f}, n={int(stats['count'])})")
    else:
         report_lines.append("  - Insufficient data to reliably compare effectiveness across different intervention types.")
else:
    report_lines.append(f"  - Insufficient or no extractable effect size data was found across the {total_studies_analyzed} studies to quantify overall effectiveness robustly.")
    report_lines.append(f"  - Only {num_studies_with_es} studies ({num_studies_with_es/total_studies_analyzed:.1%}) had any potential ES values.")


# --- Intervention Types Findings ---
report_lines.append("\n2. Predominant Intervention Types:")
if not intervention_counts.empty:
    top_intervention_type = intervention_counts.index[0]
    top_intervention_count = intervention_counts.iloc[0]
    top_intervention_percentage = (top_intervention_count / total_studies_analyzed) * 100
    report_lines.append(f"  - The most frequently studied intervention type was '{top_intervention_type}' ({top_intervention_count} studies, {top_intervention_percentage:.1f}%).")
    report_lines.append(f"  - A total of {len(intervention_counts)} distinct intervention categories were identified.")
    # Mention least common types if space/interest
    least_common = intervention_counts.tail(min(3, len(intervention_counts))).index.tolist()
    report_lines.append(f"  - Less common intervention types included: {', '.join(least_common)}.")
else:
     report_lines.append("  - Intervention types could not be determined (missing data or categorization failed).")


# --- Mathematical Domain Findings ---
report_lines.append("\n3. Mathematical Domains Focused Upon:")
if not domain_df.empty:
    top_domain = domain_df.iloc[0]['Domain']
    top_domain_count = domain_df.iloc[0]['Frequency']
    top_domain_percentage = (top_domain_count / total_studies_analyzed) * 100 # % of studies covering this domain
    report_lines.append(f"  - The most frequently addressed mathematical domain was '{top_domain}' (covered in {top_domain_count} studies, {top_domain_percentage:.1f}%).")

    # Identify less studied domains (potential gaps)
    bottom_domains = domain_df.tail(max(0, min(3, len(domain_df)-1)))['Domain'].tolist() # Avoid repeating top if only 1-2 domains
    if bottom_domains:
         report_lines.append(f"  - Domains less frequently covered included: {', '.join(bottom_domains)}.")
else:
    report_lines.append("  - Mathematical domains could not be determined (missing data or identification failed).")


# --- Key Network Relationship Findings ---
report_lines.append("\n4. Intervention-Domain Relationships:")
if not pair_frequencies.empty:
    top_pair = pair_frequencies.index[0]
    top_pair_count = pair_frequencies.iloc[0]
    report_lines.append(f"  - The strongest co-occurrence was between '{top_pair[0]}' interventions and the '{top_pair[1]}' domain ({top_pair_count} studies).")

    # Identify potential gaps from network analysis (e.g., domains with few intervention links)
    if G.number_of_nodes() > 0:
        domain_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'Domain']
        if domain_nodes:
            domain_degrees = {n: G.degree(n) for n in domain_nodes}
            sorted_domain_degrees = sorted(domain_degrees.items(), key=lambda item: item[1]) # Sort by degree ascending
            least_connected_domains = [d[0] for d in sorted_domain_degrees[:min(3, len(sorted_domain_degrees))] if d[1] <= 1] # Domains connected to 0 or 1 intervention types
            if least_connected_domains:
                report_lines.append(f"  - Potential research gaps: Domains like {', '.join(least_connected_domains)} appear to be linked with fewer types of interventions in this dataset.")
            else:
                 report_lines.append("  - All identified domains were connected to multiple intervention types.")
        else:
             report_lines.append("  - Could not analyze domain connectivity (no domain nodes in network).")
    else:
        report_lines.append("  - Network analysis was not performed or failed.")
else:
    report_lines.append("  - Insufficient data to analyze relationships between intervention types and domains.")


# --- Time Trends Summary ---
report_lines.append("\n5. Temporal Trends:")
if not yearly_metrics.empty and len(yearly_metrics) > 2:
    # Study Volume Trend (using slope calculated earlier if available)
    if 'z_counts' in locals():
         if abs(z_counts[0]) < 0.1: volume_trend = "relatively stable" # Adjust threshold as needed
         elif z_counts[0] > 0: volume_trend = "an increasing"
         else: volume_trend = "a decreasing"
         report_lines.append(f"  - Research publication volume shows {volume_trend} trend over the analyzed period (Slope ≈ {z_counts[0]:.2f} studies/year).")
    else:
         report_lines.append("  - Trend in research volume could not be determined.")

    # Effect Size Trend (using slope calculated earlier if available)
    if 'z_es' in locals() and z_es is not None: # Check if slope was calculated
        if abs(z_es[0]) < 0.005: es_trend_direction = "relatively stable"
        elif z_es[0] > 0.005: es_trend_direction = "a slight increasing"
        else: es_trend_direction = "a slight decreasing"
        report_lines.append(f"  - The average reported effect sizes show {es_trend_direction} trend over time (Slope ≈ {z_es[0]:.3f} ES units/year).")
    elif num_studies_with_es > 0:
         report_lines.append("  - Trend in effect sizes could not be reliably determined (e.g., insufficient data points).")
    #else: No need to report ES trend if no ES data exists anyway
else:
    report_lines.append("  - Insufficient data spanning multiple years to determine reliable temporal trends.")


# --- Methodological Observations ---
report_lines.append("\n6. Methodological Observations:")

# Sample Size Analysis
if 'numeric_sample_size' in df_obj1.columns and df_obj1['numeric_sample_size'].notna().sum() > 0:
    mean_sample = df_obj1['numeric_sample_size'].mean()
    median_sample = df_obj1['numeric_sample_size'].median()
    report_lines.append(f"  - Sample sizes varied, with a mean of {mean_sample:.1f} and a median of {median_sample:.0f} participants.")

    # Check for small sample sizes (e.g., < 30)
    small_sample_threshold = 30
    small_sample_count = (df_obj1['numeric_sample_size'] < small_sample_threshold).sum()
    valid_sample_count = df_obj1['numeric_sample_size'].notna().sum()
    if valid_sample_count > 0:
        small_sample_percent = (small_sample_count / valid_sample_count) * 100
        if small_sample_percent > 10: # Report if >10% are small
             report_lines.append(f"  - Approximately {small_sample_percent:.1f}% of studies with sample size data used small samples (< {small_sample_threshold} participants).")
else:
    report_lines.append("  - Sample size information was largely missing or could not be extracted numerically.")

# Effect Size Reporting Rate
es_reporting_rate = (num_studies_with_es / total_studies_analyzed) * 100 if total_studies_analyzed > 0 else 0
report_lines.append(f"  - Extractable effect sizes were reported in only {es_reporting_rate:.1f}% of the analyzed studies.")
if es_reporting_rate < 50:
    report_lines.append("    - This low reporting rate limits the ability to perform robust meta-analysis.")

# Study Design (Basic Overview - requires 'study_design' column)
if 'study_design' in df_obj1.columns:
     design_counts = df_obj1['study_design'].value_counts()
     if not design_counts.empty:
         top_design = design_counts.index[0]
         top_design_percent = (design_counts.iloc[0] / total_studies_analyzed) * 100
         report_lines.append(f"  - The most common study design mentioned appeared to be '{top_design}' ({top_design_percent:.1f}%). (Note: Requires consistent terminology in source data).")
     else:
         report_lines.append("  - Study design information was sparse or highly variable.")


# --- Implications and Recommendations ---
report_lines.append("\n7. Implications for Practice and Research:")

# Overall effectiveness conclusion (cautious if ES data is limited)
if num_studies_with_es > 0 and not np.isnan(mean_es):
    report_lines.append(f"  - The available evidence suggests interactive digital tools have a generally positive impact (average effect size: {effect_interpretation}, mean abs ES ≈ {mean_es:.3f}) on math skills/CT in primary students.")
else:
    report_lines.append("  - While many studies exist, the lack of consistent effect size reporting makes it difficult to draw strong conclusions about the overall magnitude of impact.")

# Recommendations based on findings
if not es_by_intervention_filtered.empty:
     top_performing_intervention = es_by_intervention_filtered.index[0]
     report_lines.append(f"  - Interventions like '{top_performing_intervention}' showed particularly promising results in this dataset and warrant further investigation and potential adoption.")

report_lines.append("  - Recommendations for Future Research:")
report_lines.append("    * Focus on less-studied mathematical domains (e.g., " + ", ".join(bottom_domains if 'bottom_domains' in locals() and bottom_domains else ['Domains identified as less frequent']) + ") and intervention types.")
report_lines.append("    * Improve methodological rigor, including using larger sample sizes and control groups.")
report_lines.append("    * Crucially, consistently report standardized effect sizes (e.g., Cohen's d, Hedges' g) to facilitate meta-analysis.")
report_lines.append("    * Conduct longitudinal studies to assess the long-term impact of these tools.")
if not pair_frequencies.empty and 'least_connected_domains' in locals() and least_connected_domains:
     report_lines.append(f"    * Investigate the effectiveness of various intervention types specifically for domains like {', '.join(least_connected_domains)}.")

# --- Limitations ---
report_lines.append("\n8. Limitations of This Analysis:")
report_lines.append(f"  - Based on a limited set of {total_studies_analyzed} studies found in '{data_file}'.")
report_lines.append("  - Potential publication bias (studies with positive results may be more likely published).")
report_lines.append("  - Heterogeneity in study designs, interventions, outcome measures, and reporting quality limits direct comparability.")
report_lines.append("  - Automated extraction of effect sizes, intervention types, and domains relies on keyword matching and may miss nuances or contain inaccuracies.")
report_lines.append("  - Categorization of interventions and domains involved simplification.")
report_lines.append("  - Quality appraisal of individual studies was not performed in this automated analysis.")

# --- Save the Final Report ---
report_file_path = os.path.join(output_dir, 'Objective1_Analysis_Report.txt') # More descriptive filename
try:
    with open(report_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\nComprehensive analysis report saved successfully to: {report_file_path}")
except Exception as e:
    print(f"\nError saving the final report: {str(e)}")

print("\n" + "=" * 60)
print(f"ANALYSIS COMPLETE. All generated outputs saved in the '{output_dir}' directory.")
print("=" * 60)