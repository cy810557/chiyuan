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
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *l3 = new ListNode(0);
        ListNode *head = l3;
        while(l1 || l2){
            bool up_flag=false;
            //l3->val += l1->val + l2->val;
            if(l1) {l3->val+=l1->val; l1=l1->next;}
            if(l2) {l3->val+=l2->val; l2=l2->next;}
            if (l3->val>9) {up_flag=true; l3->val-=10;}
            
            if(l1 || l2 || up_flag) {
                l3->next = new ListNode(up_flag);
                l3 = l3->next;
            }
        }
        return head; 
    }
};
