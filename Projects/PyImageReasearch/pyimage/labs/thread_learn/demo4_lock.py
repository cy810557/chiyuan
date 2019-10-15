import threading
import time
# join与lock区别：
# join()表示要堵塞主线程直到这个线程完成，并不影响子线程的同时进行，只是代表在join()后边的语句必须等待join的这个线程完成才能执行
# lock()则表示要阻止线程同时访问相同的共享数据来防止线程相互干扰，所以线程只能一个一个执行，不能同时进行。

def job1():
    global A, lock
    with lock:
        for i in range(5):
            A += 1
            time.sleep(0.1)
            print('job1: ', A)

def job2():
    global A, lock
    lock.acquire()
    for i in range(5):
        A += 10
        time.sleep(0.1)
        print('job2: ', A)
    lock.release()


if __name__ == '__main__':
    lock = threading.Lock()
    A = 0
    t1 = threading.Thread(target=job1)
    t2 = threading.Thread(target=job2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

