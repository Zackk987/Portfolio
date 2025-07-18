def fizzbuzz_list(n):
    for i in range(n):
        if i % 3 == 0 and i % 5 == 0:
            print("FizzBuzz")
        elif n % 3 == 0:
            print("Fizz")
        elif n % 5 == 0:
            print("Buzz")
        else:
            print(n)
        n += 1
    return fizzbuzz_list

n = int(input("enter a number: "))
print(fizzbuzz_list(n))