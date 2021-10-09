import numpy as np

def areConsec(array):
    cond = False
    a = np.sort(array)
    if a[0] == a[1]-1 and a[0]== a[2]-2:
        cond = True
    return cond

def main():

    n = int(1e4)
    same = 0
    consec = 0

    for i in range(n):
        array = np.random.randint(1,7,size=3)
        if array[0] == array[1] and array[0] == array[2]:
            same += 1
        elif areConsec(array):
            consec += 1

    prob_same = same/n
    prob_consecutive = consec/n

    print(f"Prob Alice = {prob_same}, Prob Bob = {prob_consecutive}")

main()
