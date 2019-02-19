# Tyson Cross       1239448
# Michael Nortje    1389486 
# Josh Isserow      675720

from multiprocessing import Process

def hello():
    print("hello world")

if __name__ == "__main__":
    for num in range(10):
        proc = Process(target=hello)
        proc.start()
        proc.join()