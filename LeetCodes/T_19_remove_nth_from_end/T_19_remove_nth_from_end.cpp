#include "ListNode.h"
class Solution {
public:
    ListNode<int>* removeNthFromEnd(ListNode<int>* head, int n) {
        ListNode<int>* p_after(head);
        ListNode<int>* p_before(head);
        while(0<n--){p_after = p_after->next;} 
        while(!p_after->next)
        {
            p_after = p_after->next;
            p_before = p_before->next;
        }
        if(n==1)  p_before->next=NULL;
        else p_before->next = p_before->next->next; 
    }
};