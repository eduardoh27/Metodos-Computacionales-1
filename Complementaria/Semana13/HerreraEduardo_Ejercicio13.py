import numpy

def main():
    
    medicionesA = [6.24, 6.31, 6.28, 6.30, 6.25, 6.26, 6.24, 6.29, 6.22, 6.28]
    medicionesB = [6.27, 6.25, 6.33, 6.27, 6.24, 6.31, 6.28, 6.29, 6.34, 6.27]
    
    p_value = 0
    alpha = 0.05
    conclusion = 'conclusion'
    
    print(f"p value = {p_value}, la conclusion es: {conclusion}")

if __name__ == '__main__':
    main()