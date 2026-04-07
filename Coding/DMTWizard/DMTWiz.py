#%%
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter 
#%%
def get_probability(module: str, dmt: str, data) -> float:
    """Return probability of selecting given DMT within a module."""
    if module not in data:
        raise ValueError(f"Module '{module}' not found.")
    
    module_counts = data[module]
    total = sum(module_counts.values())
    
    count = module_counts.get(dmt, 0)
    probability = count / total if total > 0 else 0
    
    return round(probability,5)

def sequential_probabilities(weights):
    """Compute probability that each index is the first success."""
    total = sum(weights)
    probs = [w / total for w in weights]
    
    results = []
    not_happened_yet = 1.0  # initial counter (probability all previous events not happened)
    
    for p in probs:
        prob_here = not_happened_yet * p
        results.append(prob_here)
        not_happened_yet *= (1 - p)  # update for next round
    
    return results

DMTList=[]
numbers = list(range(0, 6))
permutations = list(itertools.permutations(numbers))
moduleTuple=('Structures','Mechanisms','Dynamics','Instrumentation','Thermofluids','Materials')
# Load the CSV file
file_path = "DMT Titles Sorted.csv"  
df = pd.read_csv(file_path)
# Prepare byDMT dictionary
byDMT = {}
byModule={}
moduleList=[]
# Loop through each DMT group (DMT1, DMT2, ...)
for col in df.columns:
    # Select only columns belonging to this DMT
    modules=df[col]
    prefix=col.split('.')[0]
    if prefix in byDMT:
        for module in modules:
            if module not in byDMT[prefix]:
                byDMT[prefix][module]=1
            else:
                byDMT[prefix][module]+=1
    else:
        byDMT[prefix]={module:1 for module in modules}
    for module in modules:
        if module not in byModule:
            byModule[module]=[prefix]
        else: #module is in as a key one element alr exists
            byModule[module]+=[prefix]
    DMTList.append(prefix)
    moduleList=moduleList+list(modules)
byModule={module:dict(Counter(DMTFreq)) for module,DMTFreq in byModule.items()}
print(byModule)
moduleList=list(set(moduleList))
DMTList=list(set(DMTList))
modulePrioWeights=sequential_probabilities([6,5,4,3,2,1])
output={}
for permutation in permutations: #module combination
    module_perm=[moduleTuple[perm] for perm in permutation]
    output[permutation]={}
    for DMT in DMTList:
        P_DMT=0
        counter=0
        for module in module_perm:
            happening=modulePrioWeights[counter] #module gets chosen
            P_DMT+=happening*get_probability(module,DMT,byModule)
            counter+=1
        output[permutation][DMT]=P_DMT
print(output)
# %%
condition = ["DMT10", "DMT4", "DMT5",'DMT6','DMT11','DMT3','DMT9','DMT12','DMT15','DMT13','DMT14','DMT8','DMT1','DMT2','DMT7']

suboptimal_perms = []

for perm, dmt_probs in output.items():
    # Count satisfied consecutive inequalities
    satisfied_count = 0
    for i in range(len(condition) - 1):
        if dmt_probs.get(condition[i], 0) > dmt_probs.get(condition[i+1], 0):
            satisfied_count += 1
    
    if satisfied_count > 0:  # ignore permutations with 0 satisfaction
        module_order = [moduleTuple[i] for i in perm]
        suboptimal_perms.append(
            {
                "Permutation": module_order,
                "SatisfiedPairs": satisfied_count,
                **{dmt: round(100*dmt_probs[dmt],3) for dmt in condition}
            }
        )

# Sort by number of satisfied pairs (descending)
suboptimal_perms.sort(key=lambda x: x["DMT12"], reverse=True)

# Show top 5 closest permutations
for mp in suboptimal_perms[:5]:
    print(mp)

# Save all to CSV
df_subopt = pd.DataFrame(suboptimal_perms)
df_subopt.to_csv("suboptimal_permutations.csv", index=False)
print(f"Saved {len(suboptimal_perms)} suboptimal permutations to 'suboptimal_permutations.csv'")

# %%
# %%
# Compute skewness/front-loaded score for each permutation
suboptimal_perms_skew = []
for perm_dict in suboptimal_perms:
    module_order = perm_dict["Permutation"]
    dmt_probs = [perm_dict[dmt] for dmt in condition]
    
    # Split into first half vs second half
    n = len(dmt_probs)
    first_half = dmt_probs[:n//2]
    second_half = dmt_probs[n//2:]
    
    # Skewness metric: fraction of total probability in first half
    skew_score = sum(first_half) / sum(dmt_probs) if sum(dmt_probs) > 0 else 0
    
    suboptimal_perms_skew.append({
        "Permutation": module_order,
        "SkewScore": round(skew_score,3),
        "SatisfiedPairs": perm_dict["SatisfiedPairs"],
        **{dmt: perm_dict[dmt] for dmt in condition}
    })

# Sort by skewness descending (front-loaded permutations first)
suboptimal_perms_skew.sort(key=lambda x: x["SkewScore"], reverse=True)

# Save sorted CSV
df_skew = pd.DataFrame(suboptimal_perms_skew)
df_skew.to_csv("suboptimal_permutations_skew.csv", index=False)
print(f"Saved {len(suboptimal_perms_skew)} permutations sorted by skewness to 'suboptimal_permutations_skew.csv'")

# Optional: plot the distributions again (front-loaded order)
df_long_skew = df_skew.melt(id_vars=["Permutation"], value_vars=condition,
                            var_name="DMT", value_name="Probability")

plt.figure(figsize=(14, 6))
sns.boxplot(x="DMT", y="Probability", data=df_long_skew, palette="viridis")
sns.stripplot(x="DMT", y="Probability", data=df_long_skew, color="red", alpha=0.3, jitter=True)

plt.title("Distribution of DMT Probabilities Across Permutations (Sorted by Skewness)")
plt.ylabel("Probability (%)")
plt.xlabel("DMT")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
