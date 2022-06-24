
def binary2decimalism(binary_code):
    ans = 0
    for i in binary_code:
        ans<<=1
        ans+=i
    return ans