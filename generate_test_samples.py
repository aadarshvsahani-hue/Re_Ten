import json
from Dataset import generate_dataset

# Generate a few sample users for testing
df_samples = generate_dataset(n=5, seed=123)

# Convert to list of dicts, excluding 'retained' as it's the target
samples = []
for _, row in df_samples.iterrows():
    sample = row.drop('retained').to_dict()
    samples.append(sample)

# Save to JSON file
with open('test_samples.json', 'w') as f:
    json.dump(samples, f, indent=2)

print("Generated 5 test samples and saved to test_samples.json")
print("Sample input format:")
print(json.dumps(samples[0], indent=2))