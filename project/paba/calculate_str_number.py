import re


def calculate_str(char_str):
    #exact number
    numbers = re.findall(r'\d', char_str)
    # calculate len
    total_length = sum(len(num) for num in numbers)
    return total_length

if __name__ == "__main__":

    A = "[2,3]"
    total_length=calculate_str(A)
    print(total_length)  # 输出: 2