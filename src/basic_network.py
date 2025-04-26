def relu(x):
    return max(0, x)  


animals = [
    [1, 7],   
    [7, -1],  
    [0, 10],  
]


w = [3, -8]  


b = 8  


for x in animals:
    y = relu(w[0] * x[0] + w[1] * x[1] + b)  
    
    
    if y > 5:
        print(f"Animal {x} → It's a CAT! 🐱")
    else:
        print(f"Animal {x} → It's a DOG! 🐶")

    print(f"Output score: {y}\n")  
