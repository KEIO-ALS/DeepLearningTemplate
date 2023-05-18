from data.dataset_utils import load_addition, decode_addition


trainloader, testloader = load_addition()
for data in trainloader:
    x,y = data
    print(x)
    print(y)
    print("decode")
    print(decode_addition(x))
    print(decode_addition(y))
    break