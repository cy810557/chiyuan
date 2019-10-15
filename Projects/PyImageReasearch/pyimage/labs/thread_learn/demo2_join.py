import threading
import time

def thread_job():
    print("T1 start")
    for i in range(10):
        time.sleep(0.1)
    print("T1 finished")
    
def T2_job():
    print("T2 start")
    for i in range(20):
        time.sleep(0.1)
    print("T2 finished")
 

def main():
    added_thread = threading.Thread(target=thread_job, name="T1")
    thread2  = threading.Thread(target=T2_job, name="T2")
    added_thread.start()
    added_thread.join()  # 保证当前thread执行完毕之后控制权再转移给其他进程
    thread2.start()
    thread2.join()
    print("All done")



if __name__ == '__main__':
    main()
