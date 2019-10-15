import threading
import time
from queue import Queue

def process(lst, q):
    for i in range(len(lst)):
        lst[i] = lst[i]**2
    #lst = [x**2 for x in lst]
    # return lst   # 作为target的函数不支持返回值操作。故需用到queue
    q.put(lst)
    
    
def multithreading():
    threads = []
    q = Queue()
    for i in range(4):
        t = threading.Thread(target=process, args=(data[i], q))
        t.start()
        t.join()
        threads.append(t)

    results = []
    for _ in range(4):
        results.append(q.get())  # 先进先出
    print(results)
    

if __name__ == '__main__':
    data = [[1,1,1], [2,2,2], [3,3,3], [4,4,4]]
    multithreading()
