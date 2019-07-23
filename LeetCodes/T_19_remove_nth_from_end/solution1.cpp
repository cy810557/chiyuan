#include<iostream>
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* p_after(head);
        ListNode* dummy = new ListNode(0);
        dummy->next = head;
        ListNode* p_before(dummy);
        if(head->next==NULL || head==NULL)  {return NULL;} 
        while(0<n--){p_after = p_after->next;} 
        while(p_after)
        {
            p_after = p_after->next;
            p_before = p_before->next;
        }
        p_before->next = p_before->next->next; 
        return dummy->next;
    }
};
