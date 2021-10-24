import numpy as np
from tqdm import tqdm

def punto25():

    Ntrials = int(1e5)

    results = np.zeros((Ntrials,5))

    for i in tqdm(range(Ntrials)):
        d1 = np.random.randint(1,7)
        d2 = np.random.randint(1,7)
        d3 = np.random.randint(1,7)
        d4 = np.random.randint(1,7)
        d5 = np.random.randint(1,7)
        results[i] = [d1,d2,d3,d4,d5]

    it1 = 0
    it2 = 0

    for R in tqdm(results):
        
        R.sort()
        pares = 0

        for i in range(len(R)-1):

            if R[i] == R[i+1]:
                
                a = R[i]

                if a not in R[i+2:] and a not in R[:i]:
                    
                    pares += 1

                    if len(R[i+2:]) == 3 and R[i+2:][-1] == R[i+2:][0]:

                        pares -= 1
                    
                    elif len(R[:i]) == 3 and R[:i][-1] == R[:i][0]:

                        pares -= 1

        if pares == 1:
            it1 += 1
        elif pares == 2:
            it2 += 1

    print(it1/Ntrials, 25./54.)
    print(it2/Ntrials, 25./108.)

def main(): 
    punto25()



if __name__ == '__main__':
    main()
