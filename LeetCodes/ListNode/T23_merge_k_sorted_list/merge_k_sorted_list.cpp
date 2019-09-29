#include "../listnode.h"
ListNode<int>* mergeKLists(vector<ListNode<int>*>& lists){
    ListNode<int>* head = new ListNode<int>(0);
    return head;
}

int main(){
    vector<ListNode<int>*> heads;
    vector<int> v1 = {1, 4, 5};
    heads.push_back(v1);
    
    /**
    heads.push_back(create_listnode(vector<int>({1, 4, 5})));
    heads.push_back(create_listnode(vector<int>({1, 3, 4})));
    heads.push_back(create_listnode(vector<int>({2, 6})));
    **/
    for (auto x:heads) print(x)<<endl;
}
