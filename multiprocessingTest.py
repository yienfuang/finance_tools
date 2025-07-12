from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3, 4, 5]))
        # Output: [1, 4, 9, 16, 25]



import pandas as pd
import os

endPortSize = []
for f in os.listdir("D:/"):
    if f.startswith("portfolioSizeNoSL"):
        df = pd.read_parquet(f"D:/{f}")
        dfToAttach = df.tail(1).reset_index(drop=True)
        dfToAttach["fileName"] = f
        endPortSize.append(dfToAttach)

endPortSize = pd.concat(endPortSize, ignore_index=True)
endPortSize.sort_values("portNoSL", ascending=False, inplace=True, ignore_index=True)