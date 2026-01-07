"""
BioInsight Lite - Database Setup & Data Loader
Handles PostgreSQL database restoration and initial data extraction
"""

import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import subprocess
import os
from pathlib import Path

class ChEMBLDataLoader:
    def __init__(self, db_config):
        """
        Initialize database connection
        
        Args:
            db_config: dict with keys: host, port, database, user, password
        """
        self.db_config = db_config
        self.engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
    
    @staticmethod
    def restore_postgres_db(tar_path, db_name, user='postgres'):
        """
        Restore PostgreSQL database from tar.gz file
        
        Args:
            tar_path: Path to chembl_36_postgresql.tar.gz
            db_name: Target database name
            user: PostgreSQL user
        """
        print(f"Extracting {tar_path}...")
        extract_dir = Path(tar_path).parent / "chembl_extract"
        extract_dir.mkdir(exist_ok=True)
        
        # Extract tar.gz
        subprocess.run(["tar", "-xzf", tar_path, "-C", str(extract_dir)], check=True)
        
        # Create database
        subprocess.run([
            "psql", "-U", user, "-c", f"DROP DATABASE IF EXISTS {db_name};"
        ])
        subprocess.run([
            "psql", "-U", user, "-c", f"CREATE DATABASE {db_name};"
        ], check=True)
        
        # Restore from dump
        dump_file = extract_dir / "chembl_36" / "chembl_36_postgresql.dmp"
        subprocess.run([
            "pg_restore", "-U", user, "-d", db_name, str(dump_file)
        ], check=True)
        
        print(f"Database {db_name} restored successfully!")
    
    def load_bioactivity_data(self, limit=None):
        """
        Load bioactivity data with compound and target information
        
        Returns:
            pandas.DataFrame with merged data
        """
        query = """
        SELECT 
            a.activity_id,
            a.molregno,
            a.assay_id,
            a.doc_id,
            a.standard_type,
            a.standard_relation,
            a.standard_value,
            a.standard_units,
            a.pchembl_value,
            a.activity_comment,
            a.data_validity_comment,
            
            -- Molecule properties
            m.chembl_id as compound_chembl_id,
            m.pref_name as compound_name,
            m.max_phase,
            m.molecule_type,
            m.first_approval,
            
            -- Compound properties
            cp.mw_freebase,
            cp.alogp,
            cp.hba,
            cp.hbd,
            cp.psa,
            cp.rtb,
            cp.ro3_pass,
            cp.num_ro5_violations,
            cp.molecular_species,
            cp.full_mwt,
            cp.aromatic_rings,
            cp.heavy_atoms,
            cp.num_alerts,
            
            -- Assay information
            ass.assay_type,
            ass.assay_organism,
            ass.assay_tissue,
            ass.assay_cell_type,
            ass.assay_subcellular_fraction,
            
            -- Target information
            t.chembl_id as target_chembl_id,
            t.pref_name as target_name,
            t.target_type,
            t.organism as target_organism
            
        FROM activities a
        LEFT JOIN molecule_dictionary m ON a.molregno = m.molregno
        LEFT JOIN compound_properties cp ON m.molregno = cp.molregno
        LEFT JOIN assays ass ON a.assay_id = ass.assay_id
        LEFT JOIN target_dictionary t ON ass.tid = t.tid
        
        WHERE a.standard_value IS NOT NULL
        AND a.standard_type IN ('IC50', 'EC50', 'Ki', 'Kd', 'Potency')
        AND cp.mw_freebase IS NOT NULL
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        print("Loading bioactivity data...")
        df = pd.read_sql(query, self.engine)
        print(f"Loaded {len(df)} records")
        return df
    
    def get_table_stats(self):
        """Get row counts for key tables"""
        tables = ['activities', 'molecule_dictionary', 'target_dictionary', 
                  'assays', 'compound_properties']
        stats = {}
        
        for table in tables:
            query = f"SELECT COUNT(*) as count FROM {table}"
            result = pd.read_sql(query, self.engine)
            stats[table] = result['count'].iloc[0]
        
        return stats
    
    def search_compounds(self, search_params):
        """
        Search compounds based on various criteria
        
        Args:
            search_params: dict with search criteria
                - compound_name: str
                - target_name: str
                - min_mw: float
                - max_mw: float
                - activity_type: str
                - max_phase: int
        """
        conditions = ["1=1"]
        
        if search_params.get('compound_name'):
            conditions.append(f"m.pref_name ILIKE '%{search_params['compound_name']}%'")
        
        if search_params.get('target_name'):
            conditions.append(f"t.pref_name ILIKE '%{search_params['target_name']}%'")
        
        if search_params.get('min_mw'):
            conditions.append(f"cp.mw_freebase >= {search_params['min_mw']}")
        
        if search_params.get('max_mw'):
            conditions.append(f"cp.mw_freebase <= {search_params['max_mw']}")
        
        if search_params.get('activity_type'):
            conditions.append(f"a.standard_type = '{search_params['activity_type']}'")
        
        if search_params.get('max_phase') is not None:
            conditions.append(f"m.max_phase >= {search_params['max_phase']}")
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
        SELECT DISTINCT
            m.chembl_id,
            m.pref_name,
            m.max_phase,
            cp.mw_freebase,
            cp.alogp,
            cp.num_ro5_violations,
            t.pref_name as target_name,
            a.standard_type,
            a.standard_value,
            a.pchembl_value
        FROM molecule_dictionary m
        LEFT JOIN compound_properties cp ON m.molregno = cp.molregno
        LEFT JOIN activities a ON m.molregno = a.molregno
        LEFT JOIN assays ass ON a.assay_id = ass.assay_id
        LEFT JOIN target_dictionary t ON ass.tid = t.tid
        WHERE {where_clause}
        LIMIT 100
        """
        
        return pd.read_sql(query, self.engine)


# Example usage
if __name__ == "__main__":
    # Configuration
    DB_CONFIG = {
        'host': 'localhost',
        'port': 5432,
        'database': 'chembl_36',
        'user': 'postgres',
        'password': 'your_password'  # Use environment variable in production
    }
    
    # Initialize loader
    loader = ChEMBLDataLoader(DB_CONFIG)
    
    # Get table statistics
    stats = loader.get_table_stats()
    print("\nTable Statistics:")
    for table, count in stats.items():
        print(f"  {table}: {count:,} rows")
    
    # Load bioactivity data (sample)
    df = loader.load_bioactivity_data(limit=10000)
    print(f"\nLoaded bioactivity data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Save to CSV for faster subsequent loads
    df.to_csv('bioactivity_data.csv', index=False)
    print("Saved to bioactivity_data.csv")