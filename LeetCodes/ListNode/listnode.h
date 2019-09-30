#include<iostream>
#include<vector>
using namespace std;

#ifndef LISTNODE_
#define LISTNODE_
template<class T>
struct ListNode{
    T val;
    ListNode *next;
    ListNode(T x) : val(x), next(NULL) {}
    ListNode() : next(NULL) {}
    ListNode* operator++();
    ListNode* operator++(int n);
};

template<class T>
ListNode<T> *create_listnode(const vector<T>& vec){
    ListNode<T> *head = new ListNode<T>();
    ListNode<T> *curr = head;
    for (auto it=vec.begin();it!=vec.end()-1;++it){
        curr->val = *it;
        curr->next = new ListNode<T>();
        curr = curr -> next;
    }
    curr->val = *(vec.end()-1);
    return head;
}

template<class T>
void print(ListNode<T> *head){
    cout<<"{ ";
    while(head){
        cout<<head->val<<" -> ";
        head=head->next;
    }
    cout<<"NULL }"<<endl;
}

//函数有问题
template<class T>
ListNode<T>* ListNode<T>::operator++(){
    cout<<"DEBUG: this->val = "<<this->val<<endl;
    if(this!=NULL) this = this->next;
    else cout<<"[WARN] End of list reached."<<endl;
    return this;
}

//函数有问题
template<class T>
ListNode<T>* ListNode<T>::operator++(int n){
    ListNode<T>* temp = this;
    cout<<"DEBUG: this->val = "<<this->val<<endl;
    if(this!=NULL) this = this->next;
    else cout<<"[WARN] End of list reached."<<endl;
    return temp;
}

#endif

