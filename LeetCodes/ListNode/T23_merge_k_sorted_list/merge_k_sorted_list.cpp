#include "../listnode.h"

ListNode<int>* merge2NodeLists(ListNode<int>* head1, ListNode<int>* head2){
    ListNode<int>* head = new ListNode<int>(0);
    ListNode<int>* curr = head;
    while(head1 && head2){
        cout<<"DEBUG: head1->val="<<head1->val
            <<"; head2->val="<<head2->val<<" ";
        if(head1->val <= head2->val){
            curr -> next = new ListNode<int>(head1->val);
            head1 = head1 -> next;
        } else {
            curr -> next = new ListNode<int>(head2->val);
            head2 = head2 -> next;
        }
        curr = curr -> next;
        print(head);
    }
    while(head1){
        curr -> next = new ListNode<int>(head1->val);
        head1 = head1 -> next;
        curr = curr -> next;
    }
    while(head2){
        curr -> next = new ListNode<int>(head2->val);
        head2 = head2 -> next;
        curr = curr -> next;
    }
    print(head);
    return head->next;
} 

ListNode<int>* mergeKLists(vector<ListNode<int>*>& lists){
    //ListNode<int>* head = new ListNode<int>(0);
    int amount = lists.size();
    int interval = 1;
    while(interval<amount){
        for (int i=0;i<amount-interval;i+=interval)
            lists[i] = merge2NodeLists(lists[i], lists[i+interval]);
        interval *= 2;
    }
    return lists[0];
}

int main(){
    vector<ListNode<int>*> heads;
    heads.push_back(create_listnode(vector<int>({1, 4, 5})));
    heads.push_back(create_listnode(vector<int>({1, 3, 4})));
    heads.push_back(create_listnode(vector<int>({2, 6})));
    cout<<" print vector of listnodes: "<<endl;
    for (auto x:heads) print(x); 
    cout<<" print merge2 result: "<<endl;
    auto rst_head = merge2NodeLists(heads[0], heads[1]);
    print(rst_head);
}
