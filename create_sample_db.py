"""
Create sample bioactivity data for testing BioInsight Lite
Run this to quickly generate test data without database setup
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 10000

print("Generating sample bioactivity data...")

# Generate sample data
data = {
    # Activity identifiers
    'activity_id': range(1, n_samples + 1),
    'molregno': np.random.randint(1, 5000, n_samples),
    'assay_id': np.random.randint(1, 1000, n_samples),
    'doc_id': np.random.randint(1, 500, n_samples),
    
    # Activity measurements
    'standard_type': np.random.choice(['IC50', 'EC50', 'Ki', 'Kd', 'Potency'], n_samples, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
    'standard_relation': np.random.choice(['=', '<', '>', '<=', '>='], n_samples, p=[0.7, 0.1, 0.1, 0.05, 0.05]),
    'standard_value': np.random.lognormal(3, 2, n_samples),  # Log-normal distribution
    'standard_units': ['nM'] * n_samples,
    'pchembl_value': np.random.uniform(4, 10, n_samples),
    'activity_comment': [None] * n_samples,
    'data_validity_comment': [None] * n_samples,
    
    # Compound information
    'compound_chembl_id': ['CHEMBL' + str(i) for i in np.random.randint(1, 5000, n_samples)],
    'compound_name': [f'Compound_{i}' if np.random.rand() > 0.3 else None for i in range(n_samples)],
    'max_phase': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.3, 0.2, 0.2, 0.2, 0.1]),
    'molecule_type': np.random.choice(['Small molecule', 'Protein', 'Antibody', 'Oligosaccharide'], 
                                     n_samples, p=[0.8, 0.1, 0.05, 0.05]),
    'first_approval': [None] * n_samples,
    
    # Molecular properties
    'mw_freebase': np.random.normal(350, 100, n_samples).clip(150, 1000),
    'alogp': np.random.normal(2.5, 1.5, n_samples).clip(-2, 8),
    'hba': np.random.randint(0, 15, n_samples),
    'hbd': np.random.randint(0, 8, n_samples),
    'psa': np.random.normal(75, 30, n_samples).clip(0, 200),
    'rtb': np.random.randint(0, 20, n_samples),
    'ro3_pass': np.random.choice(['Y', 'N'], n_samples, p=[0.7, 0.3]),
    'num_ro5_violations': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.6, 0.2, 0.1, 0.07, 0.03]),
    'molecular_species': np.random.choice(['NEUTRAL', 'ACID', 'BASE', 'ZWITTERION'], 
                                         n_samples, p=[0.5, 0.2, 0.2, 0.1]),
    'full_mwt': np.random.normal(360, 105, n_samples).clip(160, 1050),
    'aromatic_rings': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.1, 0.15, 0.3, 0.25, 0.15, 0.05]),
    'heavy_atoms': np.random.randint(10, 80, n_samples),
    'num_alerts': np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.2, 0.07, 0.03]),
    
    # Assay information
    'assay_type': np.random.choice(['B', 'F', 'A', 'P', 'U'], n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
    'assay_organism': np.random.choice(['Homo sapiens', 'Mus musculus', 'Rattus norvegicus', None], 
                                      n_samples, p=[0.6, 0.15, 0.15, 0.1]),
    'assay_tissue': [None] * (n_samples // 2) + np.random.choice(['Liver', 'Brain', 'Heart', 'Lung'], 
                                                                  n_samples // 2).tolist(),
    'assay_cell_type': [None] * int(n_samples * 0.7) + np.random.choice(['HEK293', 'CHO', 'HepG2'], 
                                                                         int(n_samples * 0.3)).tolist(),
    'assay_subcellular_fraction': [None] * n_samples,
    
    # Target information
    'target_chembl_id': ['CHEMBL' + str(i) for i in np.random.randint(200, 5000, n_samples)],
    'target_name': np.random.choice([
        'Epidermal growth factor receptor',
        'Vascular endothelial growth factor receptor 2',
        'Cyclin-dependent kinase 2',
        'Matrix metalloproteinase-9',
        'Acetylcholinesterase',
        'Dopamine D2 receptor',
        'Serotonin transporter',
        'Cannabinoid CB1 receptor',
        'Histone deacetylase 1',
        'Tyrosine-protein kinase ABL1'
    ], n_samples),
    'target_type': np.random.choice(['SINGLE PROTEIN', 'PROTEIN COMPLEX', 'PROTEIN FAMILY', 'CELL-LINE'], 
                                   n_samples, p=[0.7, 0.15, 0.1, 0.05]),
    'target_organism': np.random.choice(['Homo sapiens', 'Mus musculus', None], 
                                       n_samples, p=[0.7, 0.2, 0.1])
}

# Create DataFrame
df = pd.DataFrame(data)

# Add some realistic correlations
# Make active compounds (high pchembl_value) have better drug-like properties
active_mask = df['pchembl_value'] > 6.5
df.loc[active_mask, 'num_ro5_violations'] = np.random.choice([0, 1], active_mask.sum(), p=[0.8, 0.2])
df.loc[active_mask, 'mw_freebase'] = np.random.normal(380, 80, active_mask.sum()).clip(200, 600)

# Make some missing values more realistic
missing_cols = ['compound_name', 'activity_comment', 'assay_tissue', 'assay_cell_type', 
                'assay_subcellular_fraction', 'first_approval']
for col in missing_cols:
    if col in df.columns:
        mask = np.random.rand(len(df)) < 0.3
        df.loc[mask, col] = None

# Save to CSV
output_file = 'bioactivity_data.csv'
df.to_csv(output_file, index=False)

print(f"✅ Sample data created successfully!")
print(f"   - File: {output_file}")
print(f"   - Records: {len(df):,}")
print(f"   - Columns: {len(df.columns)}")
print(f"\nColumn names:")
for col in df.columns:
    print(f"   - {col}")

print(f"\nSummary statistics:")
print(df.describe())

print(f"\n✅ You can now run the Streamlit app!")
print(f"   streamlit run streamlit_app.py")