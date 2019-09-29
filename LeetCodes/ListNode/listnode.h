#include<iostream>
using namespace std;

#ifndef LISTNODE_
#define LISTNODE_
template<class T>
struct ListNode{
    T val;
    ListNode *next;
    ListNode(T x) : val(x), next(NULL) {}
    ListNode() : next(NULL) {}
};

template<class T>
ListNode<T> *create_listnode(const vector<T>& vec){
    ListNode<T> *head = new ListNode<T>();
    ListNode<T> *curr = head;
    for (auto x : vec){
        curr->val = x;
        curr->next = new ListNode<T>();
        curr = curr -> next;
    }
    return head;
}

template<class T>
void print(ListNode<T> *head){
    cout<<"{ ";
    while(head){
        cout<<head->val<<", ";
        head=head->next;
    }
    cout<<" }";
}
#endif

