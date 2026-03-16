import pandas as pd, glob, os

# Verzeichnis mit den einzelnen Parquet-Splits (Shards)
shard_dir = r"\features\v2\parquet"
# Zielpfad für die zusammengeführte große Parquet-Datei
big_path  = r"\features\v2\features_all.parquet"

# Alle Parquet-Split-Dateien finden, die dem Namensmuster 'features-part-*.parquet' entsprechen
files = glob.glob(os.path.join(shard_dir, "features-part-*.parquet"))

# Alle Shards einlesen und zu einem großen DataFrame zusammenführen
df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)

# Den zusammengeführten DataFrame als eine einzige Parquet-Datei speichern
df.to_parquet(big_path, index=False)
