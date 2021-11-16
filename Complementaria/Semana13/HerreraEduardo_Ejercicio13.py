import numpy as np 
from scipy.stats import norm

def main():
    
    medicionesA = [6.24, 6.31, 6.28, 6.30, 6.25, 6.26, 6.24, 6.29, 6.22, 6.28]
    medicionesB = [6.27, 6.25, 6.33, 6.27, 6.24, 6.31, 6.28, 6.29, 6.34, 6.27]
    sigma = 0.05
    n_sample = len(medicionesA)
    miu_sampleA = sum(medicionesA)/n_sample
    miu_sampleB = sum(medicionesB)/n_sample
    EP = (miu_sampleA-miu_sampleB)/np.sqrt(sigma**2/n_sample + sigma**2/n_sample)
    p_value = norm.cdf(EP)*2
    alpha = 0.05
    conclusion = ("dado que el p_value es mayor que el nivel de significancia"
                    " de 0.05, las soluciones vienen de la misma fuente.")
    print(f"p value = {p_value}, la conclusion es: {conclusion}")
    

if __name__ == '__main__':
    main()
    