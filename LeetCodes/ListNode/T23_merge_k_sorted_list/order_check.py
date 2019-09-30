#amount = len(lists)
amount = 8;
interval = 1
while interval < amount:
    for i in range(0, amount - interval, interval * 2):
        print("lists[{}] = self.merge2Lists(lists[{}], lists[{}])".format(i, i, i+interval))
    interval *= 2
    print()

