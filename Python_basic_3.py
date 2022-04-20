#내장 함수

x=sum([3,8])
print("내장 함수 sum의 결과는: \n", x)

y=round(7.2)
print("내장 함수 round의 결과는: \n", y)

z=range(5)
print("내장 함수 range의 결과는: \n", z)
print("range 결과 안에 들어있는 것은: \n", list(z))

l=list([2,3])
print("내장 함수 list의 결과는: \n", l)

#user-define 함수 (Module)

def add(a,b):
    result = a+b
    return result

print("User-define 함수의 결과는: \n", add(1,2))