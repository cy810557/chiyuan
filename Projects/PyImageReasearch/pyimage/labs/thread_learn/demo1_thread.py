import threading

def thread_job():
    print("This is an added thread, number is %s" % threading.current_thread())
    

def main():
    print(threading.active_count())
    print("[INFO] Enumerating: ")
    print(threading.enumerate())
    print(threading.current_thread())
    print("Now creating a new thread...")
    print("[INFO] Enumerating: ")
    print(threading.enumerate())
    added_thread = threading.Thread(target=thread_job)
    added_thread.start()
    print("[INFO] Enumerating: ")
    print(threading.enumerate())



if __name__ == '__main__':
    main()
