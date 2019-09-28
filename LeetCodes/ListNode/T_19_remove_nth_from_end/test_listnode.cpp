#include "listnode.h"
#include<string>
int main(){
    vector<string> vs = {"this", "is", "a", "string"};
    ListNode<string> head = create_listnode(vs);
    return 0;
}
