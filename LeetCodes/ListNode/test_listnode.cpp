#include "listnode.h"
#include<string>
int main(){
    cout<<"test1: use vector<string> to create listnode: "<<endl;
    vector<string> vs = {"this", "is", "a", "string"};    //通过打印可以看出vs有五个元素，最后一个应该是\0
    ListNode<string>* head = create_listnode(vs);
    print(head);

    cout<<"test2: use vector<int> to create listnode: "<<endl;
    vector<int> vi = {1, 2, 3, 4, 5};
    ListNode<int>* head1 = create_listnode(vi);
    print(head1);

    cout<<"test3: operator ++. Init: head1->val="<<head1->val<<endl;
    /**
    cout<<"head1++->val="<<(head1 ++)->val<<endl;
    cout<<"head1->val = "<<head1->val<<endl;
    cout<<"head1++->val="<<(head1 ++)->val<<endl;
    cout<<"head1->val = "<<head1->val<<endl;
    cout<<"head1++->val="<<(head1 ++)->val<<endl;
    **/

    cout<<"++head1->val="<<(++ head1)->val<<endl;
    cout<<"head1->val = "<<head1->val<<endl;
    cout<<"++head1->val="<<(++ head1)->val<<endl;
    cout<<"head1->val = "<<head1->val<<endl;
    
    return 0;
}
