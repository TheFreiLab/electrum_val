import time
import pandas as pd
from electrum import calculate_fingerprint
from tqdm import tqdm

if __name__ == "__main__":
    test_df = pd.read_csv('datasets/coordnumber.csv')
    test_df = test_df.dropna()
    
    test_smiles = test_df['LigandSmiles'].tolist()
    test_metals = test_df['Metal'].tolist()

    start = time.perf_counter()
    result = [calculate_fingerprint(smiles, metal, radius=2, n_bits=512) for smiles, metal in tqdm(zip(test_smiles, test_metals), 
                                                                 total=len(test_smiles), 
                                                                 desc="Calculating fingerprints")]
    end = time.perf_counter()

    print(f"Electrum result: Calculated {len(result)} fingerprints")
    print(f"Time elapsed: {end - start:.6f} seconds")