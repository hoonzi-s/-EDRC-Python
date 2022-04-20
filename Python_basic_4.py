#if 조건문

a=True
if a:
    print("조건문이 True이다.")
else:
    print("조건문이 False이다.")

b=10
if b>5:
    print("조건문이 True이다.")
else:
    print("조건문이 False이다.")

#if 조건문 + 함수 (Module)
def example(x):
    if x>10:
        print("조건문이 True이다.")
    else:
        print("조건문이 False이다.")

example(5)